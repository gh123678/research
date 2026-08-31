"""Audit and summarize the shared finite-sample certificate experiment."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fixed_policy_finite_sample_certificate import strict_json_ready


THEOREM_ROUTES = (
    "direct_exact",
    "direct_softmax",
    "vfirst_nosplit_exact",
    "vfirst_nosplit_softmax",
)


def load_strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def record_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(task["task_index"]),
        tuple(task["spawn_key"]),
        int(task["trajectory_length"]),
        int(task["n_states"]),
        int(task["n_actions"]),
        float(task["pi_min"]),
        float(task["beta"]),
        float(task["mixing"]),
        float(task["gap_bonus"]),
    )


def compare_value(
    old: Any,
    new: Any,
    path: str,
    mismatches: list[dict[str, Any]],
    numeric_differences: list[float],
) -> None:
    if isinstance(old, dict):
        if not isinstance(new, dict):
            mismatches.append({"path": path, "old": old, "new": new})
            return
        for key, value in old.items():
            if key not in new:
                mismatches.append(
                    {"path": f"{path}.{key}", "old": value, "new": "<missing>"}
                )
            else:
                compare_value(
                    value,
                    new[key],
                    f"{path}.{key}",
                    mismatches,
                    numeric_differences,
                )
        return
    if isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            mismatches.append({"path": path, "old": old, "new": new})
            return
        for index, (old_item, new_item) in enumerate(zip(old, new, strict=True)):
            compare_value(
                old_item,
                new_item,
                f"{path}[{index}]",
                mismatches,
                numeric_differences,
            )
        return
    if (
        isinstance(old, (int, float))
        and not isinstance(old, bool)
        and isinstance(new, (int, float))
        and not isinstance(new, bool)
    ):
        difference = abs(float(old) - float(new))
        numeric_differences.append(difference)
        if not math.isclose(float(old), float(new), rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append({"path": path, "old": old, "new": new})
        return
    if old != new:
        mismatches.append({"path": path, "old": old, "new": new})


def route_regression(
    old_tasks: list[dict[str, Any]],
    new_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    old_by_key = {record_key(task): task for task in old_tasks}
    new_by_key = {record_key(task): task for task in new_tasks}
    mismatches: list[dict[str, Any]] = []
    numeric_differences: list[float] = []
    if old_by_key.keys() != new_by_key.keys():
        missing = sorted(set(old_by_key) - set(new_by_key))
        extra = sorted(set(new_by_key) - set(old_by_key))
        mismatches.append(
            {
                "path": "task_keys",
                "old_only_count": len(missing),
                "new_only_count": len(extra),
            }
        )
    for key in sorted(old_by_key.keys() & new_by_key.keys()):
        old_routes = old_by_key[key]["routes"]
        new_routes = new_by_key[key]["routes"]
        for route, old_metrics in old_routes.items():
            if route not in new_routes:
                mismatches.append(
                    {"path": f"{key}.routes.{route}", "new": "<missing>"}
                )
                continue
            compare_value(
                old_metrics,
                new_routes[route],
                f"{key}.routes.{route}",
                mismatches,
                numeric_differences,
            )
    return {
        "passed": not mismatches,
        "old_task_count": len(old_tasks),
        "new_task_count": len(new_tasks),
        "old_route_count": len(old_tasks[0]["routes"]),
        "new_route_count": len(new_tasks[0]["routes"]),
        "max_common_numeric_abs_difference": (
            0.0 if not numeric_differences else max(numeric_differences)
        ),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:20],
    }


def finite_values(items: list[float | None]) -> list[float]:
    return [float(item) for item in items if item is not None]


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def route_summary(
    tasks: list[dict[str, Any]], route: str, length: int
) -> dict[str, Any]:
    selected = [
        task["routes"][route]
        for task in tasks
        if int(task["trajectory_length"]) == length
    ]
    failures: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    for metrics in selected:
        statuses[str(metrics["certificate_status"])] += 1
        failures.update(metrics.get("certificate_failure_reasons", []))
    count = len(selected)
    return {
        "route": route,
        "trajectory_length": length,
        "tasks": count,
        "q_sup_error": describe([float(item["q_sup_error"]) for item in selected]),
        "high_probability_certificate_rate": float(
            np.mean([item["high_probability_certified"] for item in selected])
        ),
        "pathwise_bound_verified_rate": float(
            np.mean([item["pathwise_bound_verified"] for item in selected])
        ),
        "status_counts": dict(sorted(statuses.items())),
        "failure_reason_rates": {
            reason: failures[reason] / count for reason in sorted(failures)
        },
        "finite_sample_total_bound": describe(
            finite_values([item["finite_sample_total_bound"] for item in selected])
        ),
        "finite_sample_bound_to_error_ratio": describe(
            finite_values(
                [item["finite_sample_bound_to_error_ratio"] for item in selected]
            )
        ),
        "pathwise_total_bound": describe(
            finite_values([item["pathwise_total_bound"] for item in selected])
        ),
        "pathwise_bound_slack": describe(
            finite_values([item["pathwise_bound_slack"] for item in selected])
        ),
    }


def shared_event_summary(
    tasks: list[dict[str, Any]], length: int
) -> dict[str, Any]:
    events = [
        task["certificate"]["finite_sample"]["shared_event"]
        for task in tasks
        if int(task["trajectory_length"]) == length
    ]
    return {
        "trajectory_length": length,
        "tasks": len(events),
        "state_coverage_rate": float(
            np.mean([event["state_coverage_certified"] for event in events])
        ),
        "pair_coverage_rate": float(
            np.mean([event["pair_coverage_certified"] for event in events])
        ),
        "spectral_condition_rate": float(
            np.mean([event["spectral_condition_certified"] for event in events])
        ),
        "numerical_support_truncation_rate": float(
            np.mean([event["numerical_support_truncated"] for event in events])
        ),
        "u_S": describe(finite_values([event["u_S"] for event in events])),
        "u_X": describe(finite_values([event["u_X"] for event in events])),
        "epsilon_S": describe(
            finite_values([event["epsilon_S"] for event in events])
        ),
        "epsilon_X": describe(
            finite_values([event["epsilon_X"] for event in events])
        ),
    }


def paired_nosplit_summary(
    tasks: list[dict[str, Any]], length: int
) -> dict[str, Any]:
    selected = [task for task in tasks if int(task["trajectory_length"]) == length]
    differences = np.asarray(
        [
            task["routes"]["vfirst_nosplit_softmax"]["q_sup_error"]
            - task["routes"]["vfirst_nosplit_exact"]["q_sup_error"]
            for task in selected
        ],
        dtype=np.float64,
    )
    return {
        "trajectory_length": length,
        "tasks": int(differences.size),
        "softmax_minus_exact": describe(differences.tolist()),
        "softmax_better_rate": float(np.mean(differences < 0.0)),
        "equal_within_1e_12_rate": float(np.mean(np.abs(differences) <= 1e-12)),
    }


def write_plots(
    tasks: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    lengths = sorted({int(task["trajectory_length"]) for task in tasks})
    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    for route, label in (
        ("vfirst_nosplit_exact", "No-split exact"),
        ("vfirst_nosplit_softmax", "No-split softmax"),
    ):
        means = [
            np.mean(
                [
                    task["routes"][route]["q_sup_error"]
                    for task in tasks
                    if int(task["trajectory_length"]) == length
                ]
            )
            for length in lengths
        ]
        axis.plot(lengths, means, marker="o", label=label)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Trajectory transitions")
    axis.set_ylabel("Mean Q sup-norm error")
    axis.set_title("Same-sample V-first: exact versus finite softmax")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "nosplit_exact_vs_softmax.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for route in THEOREM_ROUTES:
        rows = [row for row in route_rows if row["route"] == route]
        axes[0].plot(
            lengths,
            [row["high_probability_certificate_rate"] for row in rows],
            marker="o",
            label=route,
        )
        axes[1].plot(
            lengths,
            [row["pathwise_bound_verified_rate"] for row in rows],
            marker="o",
            label=route,
        )
    for axis, title in zip(
        axes,
        ("High-probability certificate", "Observed pathwise verification"),
        strict=True,
    ):
        axis.set_xscale("log", base=2)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Trajectory transitions")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "certificate_rates.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=Path("results/fixed_policy_q_routes_crossfit"),
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=Path("results/fixed_policy_finite_sample_certificates"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old_dir = args.old_dir.resolve()
    new_dir = args.new_dir.resolve()
    old_tasks = load_strict_json(old_dir / "task_results.json")
    new_tasks = load_strict_json(new_dir / "task_results.json")
    regression = route_regression(old_tasks, new_tasks)
    lengths = sorted({int(task["trajectory_length"]) for task in new_tasks})
    route_rows = [
        route_summary(new_tasks, route, length)
        for route in THEOREM_ROUTES
        for length in lengths
    ]
    analysis = {
        "old_route_regression": regression,
        "shared_event_by_length": [
            shared_event_summary(new_tasks, length) for length in lengths
        ],
        "theorem_routes_by_length": route_rows,
        "nosplit_exact_vs_softmax_by_length": [
            paired_nosplit_summary(new_tasks, length) for length in lengths
        ],
    }
    (new_dir / "certificate_analysis.json").write_text(
        json.dumps(
            strict_json_ready(analysis),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    write_plots(new_tasks, route_rows, new_dir)
    print(json.dumps(regression, ensure_ascii=False, indent=2))
    if not regression["passed"]:
        raise AssertionError("old route regression failed")
    print("PASS finite-sample certificate analysis")


if __name__ == "__main__":
    main()
