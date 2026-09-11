"""Strict analyzer for FP-KERN-001 kernel feasibility outputs.

The analyzer is read-only with respect to evaluator artifacts except for
writing ``analysis.json`` and appending to ``checks.log``. It independently

1. regenerates every record from the serialized seed provenance and compares
   the full record (observables, route outputs, oracle audit) exactly;
2. reconstructs the V-first nuisance estimate from stored stream aggregates
   and the four route outputs from the pure module;
3. verifies summary rows, oracle separation, and cell membership;
4. computes count-bin coverage/RMSE/MAE, paired per-record Student-t
   intervals, ordering and false-improvement diagnostics, the Spearman
   distance diagnostic, and the diagnostic-policy value change;
5. applies the frozen five-item screen per family and emits the frozen
   four-way classification.

Any mismatch is an analysis failure yielding classification INVALID, never an
excluded record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import t as student_t

from evaluate_kernel_state_generalization import (
    FORMAL_PROTOCOL,
    build_record,
    count_bin,
    summarize,
    write_json,
)
from fixed_policy_finite_sample_certificate import strict_json_ready
from kernel_state_generalization import (
    OBSERVABLE_KEYS,
    PROHIBITED_INPUT_KEYS,
    ROUTES,
    ROUTE_POOL,
    ROUTE_PRIMARY,
    REASON_POLICY,
    aggregate_y_sums,
    evaluate_all_routes,
    vfirst_value_from_aggregates,
)


BIN_ORDER = ("0", "1-4", "5-16", "17+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_strict_json(path: Path) -> Any:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite constant {value} in {path}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=reject_constant,
    )


def _compare(expected: Any, observed: Any, path: str, mismatches: list[str]) -> None:
    if isinstance(expected, bool) or isinstance(observed, bool):
        if expected is not observed:
            mismatches.append(path)
        return
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if not math.isclose(
            float(expected), float(observed), rel_tol=1e-12, abs_tol=1e-12
        ):
            mismatches.append(path)
        return
    if type(expected) is not type(observed):
        mismatches.append(path)
        return
    if isinstance(expected, dict):
        if set(expected) != set(observed):
            mismatches.append(path + ".keys")
            return
        for key in expected:
            _compare(expected[key], observed[key], f"{path}.{key}", mismatches)
        return
    if isinstance(expected, list):
        if len(expected) != len(observed):
            mismatches.append(path + ".length")
            return
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            _compare(left, right, f"{path}[{index}]", mismatches)
        return
    if expected != observed:
        mismatches.append(path)


def regenerate_record(record: dict[str, Any]) -> dict[str, Any]:
    provenance = record["seed_provenance"]
    task_seed = np.random.SeedSequence(
        provenance["task_entropy"], spawn_key=provenance["task_spawn_key"]
    )
    length_seed = np.random.SeedSequence(
        provenance["task_entropy"], spawn_key=provenance["length_spawn_key"]
    )
    return build_record(
        family=record["family"],
        task_index=int(record["task_index"]),
        record_index=int(record["record_index"]),
        task_seed=task_seed,
        length_seed=length_seed,
        trajectory_length=int(record["trajectory_length"]),
        mixing=float(record["mixing"]),
        gap_bonus=float(record["gap_bonus"]),
        n_states=int(record["n_states"]),
        n_actions=int(record["n_actions"]),
        pi_min=float(record["pi_min"]),
        gamma=float(record["gamma"]),
        alpha=float(record["alpha"]),
        iterations=int(record["iterations"]),
    )


def check_oracle_separation(record: dict[str, Any]) -> None:
    inputs = record["observable_inputs"]
    unknown = set(inputs) - set(OBSERVABLE_KEYS) - {
        "value_iterations_used",
        "value_aggregates",
        "signature_aggregates",
        "target_aggregates",
    }
    if unknown or set(inputs) & PROHIBITED_INPUT_KEYS:
        raise AssertionError(f"record {record['record_index']} observable keys invalid")
    route_outputs = record["route_outputs"]
    serialized = json.dumps(route_outputs)
    for key in PROHIBITED_INPUT_KEYS:
        if f'"{key}"' in serialized:
            raise AssertionError(
                f"record {record['record_index']} route output carries oracle key {key}"
            )
    if "oracle_audit" in record.get("observable_inputs", {}):
        raise AssertionError("oracle audit leaked into observable inputs")


def reconstruct_routes(record: dict[str, Any]) -> dict[str, Any]:
    inputs = record["observable_inputs"]
    gamma = float(record["gamma"])
    reward_bound = float(inputs["reward_bound"])
    value_limit = reward_bound / (1.0 - gamma)
    aggregates = inputs["value_aggregates"]
    value, diagnostics = vfirst_value_from_aggregates(
        np.asarray(aggregates["state_counts"], dtype=np.int64),
        np.asarray(aggregates["reward_sums"], dtype=np.float64),
        np.asarray(aggregates["transition_counts"], dtype=np.float64),
        gamma,
        float(record["alpha"]),
        int(record["iterations"]),
        value_limit,
    )
    stored_value = np.asarray(inputs["value_estimate"], dtype=np.float64)
    if not np.array_equal(value, stored_value):
        raise AssertionError(
            f"record {record['record_index']} V-first reconstruction mismatch"
        )
    if int(diagnostics["iterations_used"]) != int(inputs["value_iterations_used"]):
        raise AssertionError("V-first iteration count mismatch")

    signature_y = aggregate_y_sums(
        np.asarray(inputs["signature_aggregates"]["reward_sums"]),
        np.asarray(inputs["signature_aggregates"]["successor_counts"]),
        value,
        gamma,
    )
    target_y = aggregate_y_sums(
        np.asarray(inputs["target_aggregates"]["reward_sums"]),
        np.asarray(inputs["target_aggregates"]["successor_counts"]),
        value,
        gamma,
    )
    for name, recomputed in (("signature_sums", signature_y), ("target_sums", target_y)):
        if not np.array_equal(np.asarray(inputs[name], dtype=np.float64), recomputed):
            raise AssertionError(f"record {record['record_index']} {name} mismatch")

    pure_inputs = {
        "n_states": int(record["n_states"]),
        "n_actions": int(record["n_actions"]),
        "gamma": gamma,
        "reward_bound": reward_bound,
        "policy": np.asarray(inputs["policy"], dtype=np.float64),
        "value_estimate": stored_value,
        "signature_counts": np.asarray(inputs["signature_counts"], dtype=np.int64),
        "signature_sums": signature_y,
        "target_counts": np.asarray(inputs["target_counts"], dtype=np.int64),
        "target_sums": target_y,
    }
    if record["record_level_failure"] is not None:
        return {"skipped": record["record_level_failure"]}
    rebuilt = strict_json_ready(evaluate_all_routes(pure_inputs))
    mismatches: list[str] = []
    _compare(record["route_outputs"], rebuilt, "route_outputs", mismatches)
    if mismatches:
        raise AssertionError(
            f"record {record['record_index']} route mismatch: {mismatches[:5]}"
        )
    for route in ROUTES:
        estimates = record["route_outputs"]["routes"][route]["estimates"]
        statuses = record["diagnostic_policy"][route]["statuses"]
        for state, row in enumerate(estimates):
            expected_status = (
                "updated" if all(value is not None for value in row) else REASON_POLICY
            )
            if statuses[state] != expected_status:
                raise AssertionError(
                    f"record {record['record_index']} diagnostic status mismatch"
                )
    return rebuilt


def paired_interval(differences: list[float]) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    n = int(values.size)
    if n < 2:
        return {
            "n": n,
            "mean": float(values.mean()) if n == 1 else None,
            "ci95_low": None,
            "ci95_high": None,
            "excludes_zero": None,
            "available": False,
        }
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, n - 1) * values.std(ddof=1) / math.sqrt(n))
    low, high = mean - half, mean + half
    return {
        "n": n,
        "mean": mean,
        "ci95_low": low,
        "ci95_high": high,
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "available": True,
    }


def _estimates_matrix(record: dict[str, Any], route: str) -> np.ndarray:
    return np.asarray(
        [
            [np.nan if value is None else float(value) for value in row]
            for row in record["route_outputs"]["routes"][route]["estimates"]
        ],
        dtype=np.float64,
    )


def family_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    n_states = int(records[0]["n_states"])
    n_actions = int(records[0]["n_actions"])

    coverage: dict[str, dict[str, dict[str, int]]] = {
        route: {bin_label: {"finite": 0, "total": 0} for bin_label in BIN_ORDER}
        for route in ROUTES
    }
    squared_error: dict[str, dict[str, list[float]]] = {
        route: {bin_label: [] for bin_label in BIN_ORDER} for route in ROUTES
    }
    absolute_error: dict[str, dict[str, list[float]]] = {
        route: {bin_label: [] for bin_label in BIN_ORDER} for route in ROUTES
    }
    eligible_zero = 0
    eligible_zero_covered = 0
    paired_rmse: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    paired_excluded: dict[tuple[str, str], int] = defaultdict(int)
    top_action_pairs: list[tuple[float, float]] = []
    top_action_excluded = 0
    false_improvement: dict[str, list[int]] = {route: [0, 0] for route in ROUTES}
    policy_deltas: dict[str, list[dict[str, float]]] = {route: [] for route in ROUTES}
    spearman_values: list[float] = []

    for record in records:
        counts = np.asarray(record["observable_inputs"]["target_counts"], dtype=np.int64)
        q_true = np.asarray(record["oracle_audit"]["q_pi"], dtype=np.float64)
        estimates = {route: _estimates_matrix(record, route) for route in ROUTES}
        primary_out = record["route_outputs"]["primary"]
        gamma = float(record["gamma"])
        transition = np.asarray(record["oracle_audit"]["transition"], dtype=np.float64)
        reward = np.asarray(record["oracle_audit"]["reward"], dtype=np.float64)
        policy = np.asarray(record["observable_inputs"]["policy"], dtype=np.float64)
        v_pi = np.asarray(record["oracle_audit"]["v_pi"], dtype=np.float64)
        mu_state = np.asarray(record["oracle_audit"]["mu_state"], dtype=np.float64)

        for route in ROUTES:
            est = estimates[route]
            for s in range(n_states):
                for a in range(n_actions):
                    bin_label = count_bin(int(counts[s, a]))
                    coverage[route][bin_label]["total"] += 1
                    if math.isfinite(est[s, a]):
                        coverage[route][bin_label]["finite"] += 1
                        squared_error[route][bin_label].append(
                            float((est[s, a] - q_true[s, a]) ** 2)
                        )
                        absolute_error[route][bin_label].append(
                            float(abs(est[s, a] - q_true[s, a]))
                        )

        bandwidths = primary_out["bandwidth"]
        for s in range(n_states):
            for a in range(n_actions):
                if counts[s, a] != 0:
                    continue
                common = primary_out["common_counts"][str(a)]
                has_support = any(
                    common[s][sp] >= 2 for sp in range(n_states) if sp != s
                )
                bandwidth_ok = (
                    bandwidths[a] is not None and float(bandwidths[a]) > 0.0
                )
                if has_support and bandwidth_ok:
                    eligible_zero += 1
                    if math.isfinite(estimates[ROUTE_PRIMARY][s, a]):
                        eligible_zero_covered += 1

        for bin_label, base_route in (("0", ROUTE_POOL), ("1-4", "local_unpooled")):
            base_est = estimates[base_route]
            primary_est = estimates[ROUTE_PRIMARY]
            common_errors: dict[str, list[float]] = {base_route: [], ROUTE_PRIMARY: []}
            for s in range(n_states):
                for a in range(n_actions):
                    if count_bin(int(counts[s, a])) != bin_label:
                        continue
                    if math.isfinite(base_est[s, a]) and math.isfinite(
                        primary_est[s, a]
                    ):
                        common_errors[base_route].append(
                            float((base_est[s, a] - q_true[s, a]) ** 2)
                        )
                        common_errors[ROUTE_PRIMARY].append(
                            float((primary_est[s, a] - q_true[s, a]) ** 2)
                        )
            if common_errors[base_route]:
                base_rmse = math.sqrt(float(np.mean(common_errors[base_route])))
                primary_rmse = math.sqrt(float(np.mean(common_errors[ROUTE_PRIMARY])))
                paired_rmse[(bin_label, base_route)].append((primary_rmse, base_rmse))
            else:
                paired_excluded[(bin_label, base_route)] += 1

        sparse_states = [
            s
            for s in range(n_states)
            if any(count_bin(int(counts[s, a])) in ("0", "1-4") for a in range(n_actions))
        ]
        common_complete = [
            s
            for s in sparse_states
            if np.all(np.isfinite(estimates[ROUTE_PRIMARY][s]))
            and np.all(np.isfinite(estimates[ROUTE_POOL][s]))
        ]
        if common_complete:
            primary_hits = float(
                np.mean(
                    [
                        int(np.argmax(estimates[ROUTE_PRIMARY][s]) == np.argmax(q_true[s]))
                        for s in common_complete
                    ]
                )
            )
            pool_hits = float(
                np.mean(
                    [
                        int(np.argmax(estimates[ROUTE_POOL][s]) == np.argmax(q_true[s]))
                        for s in common_complete
                    ]
                )
            )
            top_action_pairs.append((primary_hits, pool_hits))
        else:
            top_action_excluded += 1

        fi_counts: dict[str, list[int]] = {route: [0, 0] for route in (ROUTE_PRIMARY, ROUTE_POOL)}
        for s in range(n_states):
            for a in range(n_actions):
                for b in range(n_actions):
                    if a == b:
                        continue
                    if not all(
                        math.isfinite(estimates[r][s, act])
                        for r in (ROUTE_PRIMARY, ROUTE_POOL)
                        for act in (a, b)
                    ):
                        continue
                    true_diff = q_true[s, a] - q_true[s, b]
                    for route in (ROUTE_PRIMARY, ROUTE_POOL):
                        est_diff = estimates[route][s, a] - estimates[route][s, b]
                        fi_counts[route][1] += 1
                        if est_diff > 0.0 and true_diff <= 0.0:
                            fi_counts[route][0] += 1
        for route in (ROUTE_PRIMARY, ROUTE_POOL):
            false_improvement[route][0] += fi_counts[route][0]
            false_improvement[route][1] += fi_counts[route][1]

        reward_sa = np.sum(transition * reward, axis=2)
        for route in ROUTES:
            statuses = record["diagnostic_policy"][route]["statuses"]
            new_policy = policy.copy()
            est = estimates[route]
            for s in range(n_states):
                if statuses[s] == "updated":
                    greedy = int(np.argmax(est[s]))
                    row = np.full(n_actions, float(record["pi_min"]))
                    row[greedy] = 1.0 - (n_actions - 1) * float(record["pi_min"])
                    new_policy[s] = row
            p_new = np.einsum("sa,san->sn", new_policy, transition)
            reward_new = np.sum(new_policy * reward_sa, axis=1)
            v_new = np.linalg.solve(
                np.eye(n_states) - gamma * p_new, reward_new
            )
            delta = v_new - v_pi
            policy_deltas[route].append(
                {
                    "return_change": float(mu_state @ delta),
                    "min_component_change": float(np.min(delta)),
                    "max_component_change": float(np.max(delta)),
                }
            )

        for a in range(n_actions):
            common = primary_out["common_counts"][str(a)]
            distance = primary_out["distances"][str(a)]
            xs: list[float] = []
            ys: list[float] = []
            for s in range(n_states):
                for sp in range(s + 1, n_states):
                    if common[s][sp] >= 2 and distance[s][sp] is not None:
                        xs.append(float(distance[s][sp]))
                        ys.append(float(abs(q_true[s, a] - q_true[sp, a])))
            if len(xs) >= 2:
                correlation = spearmanr(xs, ys).statistic
                if math.isfinite(float(correlation)):
                    spearman_values.append(float(correlation))

    def rmse(values: list[float]) -> float | None:
        return None if not values else math.sqrt(float(np.mean(values)))

    def mae(values: list[float]) -> float | None:
        return None if not values else float(np.mean(values))

    coverage_report = {
        route: {
            bin_label: {
                "total": coverage[route][bin_label]["total"],
                "finite": coverage[route][bin_label]["finite"],
                "rate": (
                    None
                    if coverage[route][bin_label]["total"] == 0
                    else coverage[route][bin_label]["finite"]
                    / coverage[route][bin_label]["total"]
                ),
                "rmse": rmse(squared_error[route][bin_label]),
                "mae": mae(absolute_error[route][bin_label]),
            }
            for bin_label in BIN_ORDER
        }
        for route in ROUTES
    }

    screen: dict[str, Any] = {}
    zero_coverage_rate = (
        None if eligible_zero == 0 else eligible_zero_covered / eligible_zero
    )
    screen["item1_zero_coverage"] = {
        "eligible": eligible_zero,
        "covered": eligible_zero_covered,
        "rate": zero_coverage_rate,
        "threshold": 0.5,
        "passed": bool(zero_coverage_rate is not None and zero_coverage_rate >= 0.5),
    }
    for item, (bin_label, base_route) in zip(
        ("item2_zero_count_rmse", "item3_low_count_rmse"),
        (("0", ROUTE_POOL), ("1-4", "local_unpooled")),
        strict=True,
    ):
        pairs = paired_rmse[(bin_label, base_route)]
        differences = [base - primary for primary, base in pairs]
        interval = paired_interval(differences)
        primary_mean = (
            None if not pairs else float(np.mean([primary for primary, _ in pairs]))
        )
        base_mean = None if not pairs else float(np.mean([base for _, base in pairs]))
        reduction = (
            None
            if primary_mean is None or not base_mean
            else (base_mean - primary_mean) / base_mean
        )
        screen[item] = {
            "bin": bin_label,
            "baseline_route": base_route,
            "contributing_records": interval["n"],
            "excluded_records": paired_excluded[(bin_label, base_route)],
            "primary_rmse_mean": primary_mean,
            "baseline_rmse_mean": base_mean,
            "relative_reduction": reduction,
            "paired_improvement": interval,
            "passed": bool(
                interval["available"]
                and interval["excludes_zero"]
                and interval["mean"] > 0.0
                and reduction is not None
                and reduction >= 0.10
            ),
        }
    top_differences = [primary - pool for primary, pool in top_action_pairs]
    top_interval = paired_interval(top_differences)
    screen["item4_sparse_top_action"] = {
        "baseline_route": ROUTE_POOL,
        "contributing_records": top_interval["n"],
        "excluded_records": top_action_excluded,
        "primary_accuracy_mean": (
            None
            if not top_action_pairs
            else float(np.mean([primary for primary, _ in top_action_pairs]))
        ),
        "baseline_accuracy_mean": (
            None
            if not top_action_pairs
            else float(np.mean([pool for _, pool in top_action_pairs]))
        ),
        "paired_improvement": top_interval,
        "threshold": 0.05,
        "passed": bool(
            top_interval["available"]
            and top_interval["excludes_zero"]
            and top_interval["mean"] >= 0.05
        ),
    }
    primary_rate = (
        None
        if false_improvement[ROUTE_PRIMARY][1] == 0
        else false_improvement[ROUTE_PRIMARY][0] / false_improvement[ROUTE_PRIMARY][1]
    )
    pool_rate = (
        None
        if false_improvement[ROUTE_POOL][1] == 0
        else false_improvement[ROUTE_POOL][0] / false_improvement[ROUTE_POOL][1]
    )
    screen["item5_false_improvement"] = {
        "baseline_route": ROUTE_POOL,
        "primary_rate": primary_rate,
        "baseline_rate": pool_rate,
        "primary_counts": false_improvement[ROUTE_PRIMARY],
        "baseline_counts": false_improvement[ROUTE_POOL],
        "tolerance": 0.01,
        "passed": bool(
            primary_rate is not None
            and pool_rate is not None
            and primary_rate <= pool_rate + 0.01
        ),
    }
    screen_passed = all(item["passed"] for item in screen.values())

    spearman_interval = paired_interval(spearman_values)
    spearman_report = {
        "finite_records": spearman_interval["n"],
        "mean": spearman_interval["mean"],
        "ci95_low": spearman_interval["ci95_low"],
        "ci95_high": spearman_interval["ci95_high"],
        "passed": bool(
            spearman_interval["available"]
            and spearman_interval["excludes_zero"]
            and spearman_interval["mean"] is not None
            and spearman_interval["mean"] > 0.0
        ),
    }

    policy_report = {
        route: {
            "records": len(policy_deltas[route]),
            "return_change_mean": float(
                np.mean([item["return_change"] for item in policy_deltas[route]])
            ),
            "min_component_change_mean": float(
                np.mean([item["min_component_change"] for item in policy_deltas[route]])
            ),
            "nondegenerate_records": int(
                sum(
                    abs(item["return_change"]) > 0.0 or abs(item["min_component_change"]) > 0.0
                    for item in policy_deltas[route]
                )
            ),
        }
        for route in ROUTES
    }

    return {
        "records": len(records),
        "coverage": coverage_report,
        "screen": screen,
        "screen_passed": screen_passed,
        "spearman_distance_diagnostic": spearman_report,
        "diagnostic_policy": policy_report,
    }


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path, default=project / "results/FP-KERN-001/claude"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    config = load_strict_json(result_dir / "config.json")
    records = load_strict_json(result_dir / "task_results.json")
    summary = load_strict_json(result_dir / "summary.json")

    integrity_failures: list[str] = []
    classification = "INVALID"
    analysis: dict[str, Any] = {}
    try:
        expected_cells = {
            (family, length, mixing, gap)
            for family in config["families"]
            for length in config["trajectory_lengths"]
            for mixing in config["mixing"]
            for gap in config["gap_bonuses"]
        }
        observed_cells = {
            (
                record["family"],
                int(record["trajectory_length"]),
                float(record["mixing"]),
                float(record["gap_bonus"]),
            )
            for record in records
        }
        if observed_cells != expected_cells:
            raise AssertionError("cell membership mismatch")
        expected_records = len(expected_cells) * int(config["tasks_per_cell"])
        if len(records) != expected_records:
            raise AssertionError(
                f"record count {len(records)} != {expected_records}"
            )
        if config["mode"] == "formal" and len(records) != FORMAL_PROTOCOL["records"]:
            raise AssertionError("formal mode requires exactly 480 records")
        if list(config["routes"]) != list(ROUTES):
            raise AssertionError("route order changed")

        mismatches: list[str] = []
        for record in records:
            check_oracle_separation(record)
            if record["record_level_failure"] is None:
                reconstruct_routes(record)
            regenerated = strict_json_ready(regenerate_record(record))
            _compare(record, regenerated, f"record[{record['record_index']}]", mismatches)
            if len(mismatches) > 20:
                break
        if mismatches:
            raise AssertionError(
                f"seed regeneration mismatch in {len(mismatches)} fields: {mismatches[:10]}"
            )

        rebuilt_summary = strict_json_ready(summarize(records))
        summary_mismatches: list[str] = []
        _compare(summary, rebuilt_summary, "summary", summary_mismatches)
        if summary_mismatches:
            raise AssertionError(f"summary mismatch: {summary_mismatches[:10]}")

        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_family[record["family"]].append(record)
        families_report = {
            family: family_metrics(by_family[family]) for family in sorted(by_family)
        }
        hidden_passed = families_report["hidden_cluster"]["screen_passed"]
        current_passed = families_report["current_unstructured"]["screen_passed"]
        if hidden_passed and current_passed:
            classification = "GENERAL_FEASIBLE"
        elif hidden_passed:
            classification = "STRUCTURE_CONDITIONAL"
        else:
            classification = "NOT_SUPPORTED"
        analysis["families"] = families_report
    except AssertionError as error:
        integrity_failures.append(str(error))
        classification = "INVALID"

    analysis.update(
        {
            "task_id": config.get("task_id"),
            "mode": config.get("mode"),
            "record_count": len(records),
            "integrity_failures": integrity_failures,
            "classification": classification,
            "classification_rule": (
                "GENERAL_FEASIBLE if both family screens pass; "
                "STRUCTURE_CONDITIONAL if only hidden-cluster passes; "
                "NOT_SUPPORTED if hidden-cluster fails; INVALID on integrity failure"
            ),
        }
    )
    write_json(result_dir / "analysis.json", analysis)
    checks = [
        f"records={len(records)} mode={config.get('mode')}",
        f"integrity_failures={len(integrity_failures)}",
        f"classification={classification}",
    ]
    with (result_dir / "checks.log").open("a", encoding="utf-8") as handle:
        handle.write("\nANALYSIS:\n" + "\n".join(checks) + "\n")
    if integrity_failures:
        print(f"FAIL FP-KERN-001 analysis: {integrity_failures[0]}")
        return 1
    print(
        f"PASS FP-KERN-001 analysis: records={len(records)}, "
        f"classification={classification}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
