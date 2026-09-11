"""Strict reconstruction and feasibility analysis for FP-KERN-001."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import t as student_t

from evaluate_fixed_policy_q_routes import (
    grouped_exact_mean,
    iterative_state_evaluation,
    make_policy,
)
from kernel_generalization_mdps import make_environment
from kernel_state_generalization import ROUTE_NAMES, build_kernel_routes
from mdps import rollout
from verify_fixed_policy_q_routes import policy_quantities


FAMILIES = ("current_unstructured", "hidden_cluster")


def _strict_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=no_duplicates,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError("nonfinite output")
        return converted
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(type(value).__name__)


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_nested_close(observed: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise AssertionError(f"dictionary mismatch at {path}")
        for key in expected:
            _assert_nested_close(observed[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise AssertionError(f"list mismatch at {path}")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _assert_nested_close(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if observed != expected:
            raise AssertionError(f"value mismatch at {path}: {observed!r} != {expected!r}")
        return
    if isinstance(expected, int) and not isinstance(expected, bool):
        if observed != expected:
            raise AssertionError(f"integer mismatch at {path}")
        return
    if isinstance(expected, float):
        if not isinstance(observed, (int, float)) or not np.isclose(
            float(observed), expected, rtol=1e-12, atol=1e-12
        ):
            raise AssertionError(f"numeric mismatch at {path}: {observed!r} != {expected!r}")
        return
    raise TypeError(f"unsupported comparison value at {path}: {type(expected).__name__}")


def _pair_aggregates(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    value: np.ndarray,
    gamma: float,
    n_states: int,
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = rewards[1:] + gamma * value[states[1:]]
    pairs = states[:-1] * n_actions + actions[:-1]
    q_values, counts = grouped_exact_mean(targets, pairs, n_states * n_actions)
    sums = np.bincount(
        pairs, weights=targets, minlength=n_states * n_actions
    ).astype(np.float64)
    return (
        q_values.reshape(n_states, n_actions),
        counts.reshape(n_states, n_actions),
        sums.reshape(n_states, n_actions),
    )


def _route_oracle(
    mdp: dict[str, Any],
    old_value: np.ndarray,
    old_return: float,
    true_q: np.ndarray,
    route: dict[str, Any],
) -> dict[str, Any]:
    estimate = route["estimate"]
    q_error = [
        [
            None if value is None else float(value - true_q[state, action])
            for action, value in enumerate(row)
        ]
        for state, row in enumerate(estimate)
    ]
    policy = np.asarray(route["diagnostic_policy"]["policy"], dtype=np.float64)
    values = np.asarray(policy_quantities(mdp, policy)["v_pi"], dtype=np.float64)
    new_return = float(np.asarray(mdp["p0"], dtype=np.float64) @ values)
    return {
        "q_error": q_error,
        "diagnostic_value": values.tolist(),
        "componentwise_value_change": (values - old_value).tolist(),
        "diagnostic_return": new_return,
        "return_change": new_return - old_return,
    }


def reconstruct_record(record: dict[str, Any], config: dict[str, Any]) -> None:
    family = str(record["environment_family"])
    family_index = FAMILIES.index(family)
    length = int(record["trajectory_length"])
    mixing = float(record["mixing"])
    gap_bonus = float(record["gap_bonus"])
    mixing_index = list(config["mixing"]).index(mixing)
    gap_index = list(config["gap_bonuses"]).index(gap_bonus)
    components = [
        int(config["seed"]),
        family_index,
        length,
        mixing_index,
        gap_index,
        int(record["task_index"]),
    ]
    if record["seed_components"] != components:
        raise AssertionError("seed component mismatch")
    generators = [
        np.random.default_rng(child)
        for child in np.random.SeedSequence(components).spawn(5)
    ]
    mdp, hidden_audit = make_environment(
        family,
        int(config["n_states"]),
        int(config["n_actions"]),
        float(config["gamma"]),
        mixing,
        gap_bonus,
        generators[0],
    )
    policy = make_policy(
        int(config["n_states"]),
        int(config["n_actions"]),
        float(config["pi_min"]),
        generators[1],
    )
    truth = policy_quantities(mdp, policy)
    stationary = np.asarray(truth["mu_state"], dtype=np.float64)
    trajectories: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for rng in generators[2:]:
        start = int(rng.choice(int(config["n_states"]), p=stationary))
        trajectories.append(rollout(mdp, policy, start, length, rng=rng))
    reward_bound = 1.0 + gap_bonus
    value_bound = reward_bound / (1.0 - float(config["gamma"]))
    value_hat, value_diagnostics = iterative_state_evaluation(
        trajectories[0][0],
        trajectories[0][2][1:],
        int(config["n_states"]),
        float(config["gamma"]),
        float(config["alpha"]),
        int(config["iterations"]),
        None,
        value_bound,
    )
    signature_q, signature_counts, _ = _pair_aggregates(
        *trajectories[1],
        value_hat,
        float(config["gamma"]),
        int(config["n_states"]),
        int(config["n_actions"]),
    )
    _, target_counts, target_sums = _pair_aggregates(
        *trajectories[2],
        value_hat,
        float(config["gamma"]),
        int(config["n_states"]),
        int(config["n_actions"]),
    )
    observables = {
        "signature_q": signature_q.tolist(),
        "signature_counts": signature_counts.tolist(),
        "target_sums": target_sums.tolist(),
        "target_counts": target_counts.tolist(),
        "current_policy": policy.tolist(),
        "pi_min": float(config["pi_min"]),
        "value_bound": value_bound,
    }
    _assert_nested_close(record["value_estimate"], value_hat.tolist(), "value_estimate")
    _assert_nested_close(record["value_diagnostics"], value_diagnostics, "value_diagnostics")
    _assert_nested_close(record["observable_inputs"], observables, "observable_inputs")
    rebuilt_routes = build_kernel_routes(observables)
    _assert_nested_close(
        record["kernel_generalization"], rebuilt_routes, "kernel_generalization"
    )

    true_v = np.asarray(truth["v_pi"], dtype=np.float64)
    true_q = np.asarray(truth["q_pi"], dtype=np.float64)
    old_return = float(np.asarray(mdp["p0"], dtype=np.float64) @ true_v)
    oracle = {
        "true_v": true_v.tolist(),
        "true_q": true_q.tolist(),
        "old_return": old_return,
        "routes": {
            route: _route_oracle(
                mdp,
                true_v,
                old_return,
                true_q,
                rebuilt_routes["routes"][route],
            )
            for route in ROUTE_NAMES
        },
        "generator": hidden_audit,
    }
    _assert_nested_close(record["oracle_audit"], oracle, "oracle_audit")


def _estimate_array(record: dict[str, Any], route: str) -> np.ndarray:
    raw = record["kernel_generalization"]["routes"][route]["estimate"]
    return np.array(
        [[np.nan if value is None else float(value) for value in row] for row in raw],
        dtype=np.float64,
    )


def _mean_ci(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if array.size < 2:
        return {"n": int(array.size), "mean": mean, "lower": None, "upper": None}
    sem = float(np.std(array, ddof=1) / np.sqrt(array.size))
    critical = float(student_t.ppf(0.975, df=array.size - 1))
    return {
        "n": int(array.size),
        "mean": mean,
        "lower": mean - critical * sem,
        "upper": mean + critical * sem,
    }


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def _false_improvement_counts(
    true_q: np.ndarray,
    primary: np.ndarray,
    baseline: np.ndarray,
) -> tuple[int, int, int]:
    """Count the frozen one-sided false-improvement event on common pairs."""
    if true_q.shape != primary.shape or true_q.shape != baseline.shape:
        raise ValueError("false-improvement arrays must have one common shape")
    primary_false = 0
    baseline_false = 0
    comparisons = 0
    for state in range(true_q.shape[0]):
        for left in range(true_q.shape[1]):
            for right in range(left + 1, true_q.shape[1]):
                if not (
                    np.isfinite(primary[state, left])
                    and np.isfinite(primary[state, right])
                    and np.isfinite(baseline[state, left])
                    and np.isfinite(baseline[state, right])
                ):
                    continue
                true_difference = float(true_q[state, left] - true_q[state, right])
                primary_difference = float(
                    primary[state, left] - primary[state, right]
                )
                baseline_difference = float(
                    baseline[state, left] - baseline[state, right]
                )
                primary_false += int(
                    primary_difference > 0.0 and true_difference <= 0.0
                )
                baseline_false += int(
                    baseline_difference > 0.0 and true_difference <= 0.0
                )
                comparisons += 1
    return primary_false, baseline_false, comparisons


def _record_action_correlations(
    distances: list[list[list[float | None]]],
    true_q: np.ndarray,
) -> list[float]:
    """Compute one finite Spearman correlation for each eligible record/action."""
    correlations: list[float] = []
    for action in range(true_q.shape[1]):
        observed_distance: list[float] = []
        observed_difference: list[float] = []
        for left in range(true_q.shape[0]):
            for right in range(left + 1, true_q.shape[0]):
                distance = distances[action][left][right]
                if distance is None or not np.isfinite(float(distance)):
                    continue
                observed_distance.append(float(distance))
                observed_difference.append(
                    abs(float(true_q[left, action] - true_q[right, action]))
                )
        if len(observed_distance) < 3:
            continue
        correlation = float(
            spearmanr(observed_distance, observed_difference).statistic
        )
        if np.isfinite(correlation):
            correlations.append(correlation)
    return correlations


def _family_screen(records: list[dict[str, Any]]) -> dict[str, Any]:
    zero_eligible = 0
    zero_covered = 0
    zero_primary: list[float] = []
    zero_baseline: list[float] = []
    zero_improvement: list[float] = []
    sparse_primary: list[float] = []
    sparse_baseline: list[float] = []
    sparse_improvement: list[float] = []
    top_improvement: list[float] = []
    top_primary_values: list[float] = []
    top_baseline_values: list[float] = []
    primary_false = 0
    baseline_false = 0
    false_denominator = 0
    correlations: list[float] = []
    zero_empty_records = 0
    sparse_empty_records = 0
    top_empty_records = 0

    for record in records:
        counts = np.asarray(record["observable_inputs"]["target_counts"], dtype=np.int64)
        true_q = np.asarray(record["oracle_audit"]["true_q"], dtype=np.float64)
        primary = _estimate_array(record, "leave_one_action_out_kernel")
        action_pool = _estimate_array(record, "action_only_pool")
        local = _estimate_array(record, "local_unpooled")
        eligible = np.asarray(
            record["kernel_generalization"]["routes"]["leave_one_action_out_kernel"]["kernel"]["signature_eligible"],
            dtype=bool,
        )
        zero_mask_all = counts == 0
        zero_eligible += int(np.sum(zero_mask_all & eligible))
        zero_covered += int(np.sum(zero_mask_all & eligible & np.isfinite(primary)))

        zero_common = zero_mask_all & np.isfinite(primary) & np.isfinite(action_pool)
        if np.any(zero_common):
            primary_rmse = _rmse(primary[zero_common] - true_q[zero_common])
            baseline_rmse = _rmse(action_pool[zero_common] - true_q[zero_common])
            zero_primary.append(primary_rmse)
            zero_baseline.append(baseline_rmse)
            zero_improvement.append(baseline_rmse - primary_rmse)
        else:
            zero_empty_records += 1

        sparse_common = (
            (counts >= 1)
            & (counts <= 4)
            & np.isfinite(primary)
            & np.isfinite(local)
        )
        if np.any(sparse_common):
            primary_rmse = _rmse(primary[sparse_common] - true_q[sparse_common])
            baseline_rmse = _rmse(local[sparse_common] - true_q[sparse_common])
            sparse_primary.append(primary_rmse)
            sparse_baseline.append(baseline_rmse)
            sparse_improvement.append(baseline_rmse - primary_rmse)
        else:
            sparse_empty_records += 1

        sparse_states = np.any(counts <= 4, axis=1)
        complete = (
            np.all(np.isfinite(primary), axis=1)
            & np.all(np.isfinite(action_pool), axis=1)
            & sparse_states
        )
        if np.any(complete):
            truth_actions = np.argmax(true_q[complete], axis=1)
            primary_accuracy = float(
                np.mean(np.argmax(primary[complete], axis=1) == truth_actions)
            )
            baseline_accuracy = float(
                np.mean(np.argmax(action_pool[complete], axis=1) == truth_actions)
            )
            top_primary_values.append(primary_accuracy)
            top_baseline_values.append(baseline_accuracy)
            top_improvement.append(primary_accuracy - baseline_accuracy)
        else:
            top_empty_records += 1

        record_primary_false, record_baseline_false, record_comparisons = (
            _false_improvement_counts(true_q, primary, action_pool)
        )
        primary_false += record_primary_false
        baseline_false += record_baseline_false
        false_denominator += record_comparisons

        distances = record["kernel_generalization"]["routes"]["leave_one_action_out_kernel"]["kernel"]["distance_by_action"]
        correlations.extend(_record_action_correlations(distances, true_q))

    coverage = zero_covered / zero_eligible if zero_eligible else None
    zero_ci = _mean_ci(zero_improvement)
    sparse_ci = _mean_ci(sparse_improvement)
    top_ci = _mean_ci(top_improvement)
    correlation_ci = _mean_ci(correlations)
    zero_relative = (
        float(np.mean(zero_improvement)) / float(np.mean(zero_baseline))
        if zero_baseline and float(np.mean(zero_baseline)) > 0.0
        else None
    )
    sparse_relative = (
        float(np.mean(sparse_improvement)) / float(np.mean(sparse_baseline))
        if sparse_baseline and float(np.mean(sparse_baseline)) > 0.0
        else None
    )
    primary_false_rate = primary_false / false_denominator if false_denominator else None
    baseline_false_rate = baseline_false / false_denominator if false_denominator else None
    criteria = {
        "zero_eligible_coverage_at_least_50pct": coverage is not None and coverage >= 0.50,
        "zero_rmse_improves_10pct_and_ci_positive": (
            zero_relative is not None
            and zero_relative >= 0.10
            and zero_ci["lower"] is not None
            and zero_ci["lower"] > 0.0
        ),
        "sparse_rmse_improves_10pct_and_ci_positive": (
            sparse_relative is not None
            and sparse_relative >= 0.10
            and sparse_ci["lower"] is not None
            and sparse_ci["lower"] > 0.0
        ),
        "top_action_improves_5pp_and_ci_positive": (
            top_ci["mean"] is not None
            and top_ci["mean"] >= 0.05
            and top_ci["lower"] is not None
            and top_ci["lower"] > 0.0
        ),
        "false_improvement_within_1pp": (
            primary_false_rate is not None
            and baseline_false_rate is not None
            and primary_false_rate <= baseline_false_rate + 0.01
        ),
    }
    return {
        "records": len(records),
        "zero_coverage": {
            "eligible": zero_eligible,
            "covered": zero_covered,
            "rate": coverage,
        },
        "zero_rmse": {
            "primary_mean": float(np.mean(zero_primary)) if zero_primary else None,
            "baseline_mean": float(np.mean(zero_baseline)) if zero_baseline else None,
            "relative_improvement": zero_relative,
            "paired_improvement": zero_ci,
            "empty_records": zero_empty_records,
        },
        "sparse_rmse": {
            "primary_mean": float(np.mean(sparse_primary)) if sparse_primary else None,
            "baseline_mean": float(np.mean(sparse_baseline)) if sparse_baseline else None,
            "relative_improvement": sparse_relative,
            "paired_improvement": sparse_ci,
            "empty_records": sparse_empty_records,
        },
        "top_action": {
            "primary_mean": float(np.mean(top_primary_values)) if top_primary_values else None,
            "baseline_mean": float(np.mean(top_baseline_values)) if top_baseline_values else None,
            "paired_improvement": top_ci,
            "empty_records": top_empty_records,
        },
        "false_improvement": {
            "comparisons": false_denominator,
            "primary_count": primary_false,
            "baseline_count": baseline_false,
            "primary_rate": primary_false_rate,
            "baseline_rate": baseline_false_rate,
        },
        "signature_q_difference_spearman": {
            "record_action_summary": correlation_ci,
            "passes_secondary_hypothesis": (
                correlation_ci["mean"] is not None
                and correlation_ci["mean"] > 0.0
                and correlation_ci["lower"] is not None
                and correlation_ci["lower"] > 0.0
            ),
        },
        "criteria": criteria,
        "screen_pass": all(criteria.values()),
    }


def _descriptive_route_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for route in ROUTE_NAMES:
        errors_by_bin: dict[str, list[float]] = {
            "0": [],
            "1-4": [],
            "5-16": [],
            "17+": [],
        }
        estimates = 0
        total = 0
        return_changes: list[float] = []
        for record in records:
            counts = np.asarray(record["observable_inputs"]["target_counts"], dtype=np.int64)
            estimate = _estimate_array(record, route)
            truth = np.asarray(record["oracle_audit"]["true_q"], dtype=np.float64)
            finite = np.isfinite(estimate)
            estimates += int(np.sum(finite))
            total += int(finite.size)
            for state, action in np.argwhere(finite):
                count = int(counts[state, action])
                label = "0" if count == 0 else "1-4" if count <= 4 else "5-16" if count <= 16 else "17+"
                errors_by_bin[label].append(float(estimate[state, action] - truth[state, action]))
            return_changes.append(
                float(record["oracle_audit"]["routes"][route]["return_change"])
            )
        output[route] = {
            "coverage": estimates / total if total else None,
            "estimated_pairs": estimates,
            "total_pairs": total,
            "count_bins": {
                label: {
                    "n": len(values),
                    "rmse": _rmse(np.asarray(values)) if values else None,
                    "mae": float(np.mean(np.abs(values))) if values else None,
                }
                for label, values in errors_by_bin.items()
            },
            "diagnostic_return_change": _mean_ci(return_changes),
        }
    return output


def analyze(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_family = {
        family: [record for record in records if record["environment_family"] == family]
        for family in FAMILIES
    }
    screens = {family: _family_screen(selected) for family, selected in by_family.items()}
    hidden_pass = bool(screens["hidden_cluster"]["screen_pass"])
    current_pass = bool(screens["current_unstructured"]["screen_pass"])
    if not hidden_pass:
        classification = "NOT_SUPPORTED"
    elif current_pass:
        classification = "GENERAL_FEASIBLE"
    else:
        classification = "STRUCTURE_CONDITIONAL"
    summary = {
        "status": "PASS",
        "records": len(records),
        "classification": classification,
        "family_screens": screens,
        "route_metrics": {
            family: _descriptive_route_metrics(selected)
            for family, selected in by_family.items()
        },
    }
    analysis = {
        "status": "PASS",
        "record_count": len(records),
        "reconstruction_mismatches": 0,
        "classification": classification,
        "decision_rule": {
            "hidden_cluster_pass": hidden_pass,
            "current_unstructured_pass": current_pass,
        },
    }
    return summary, analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--write-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    config_path = result_dir / "config.json"
    records_path = result_dir / "task_results.json"
    environment_path = result_dir / "environment.json"
    before = {
        path.name: _sha256(path)
        for path in (config_path, records_path, environment_path)
    }
    config = _strict_load(config_path)
    records = _strict_load(records_path)
    if config["task_id"] != "FP-KERN-001" or config["routes"] != list(ROUTE_NAMES):
        raise AssertionError("config task or route mismatch")
    expected = 16 if config["mode"] == "smoke" else 480
    if not isinstance(records, list) or len(records) != expected:
        raise AssertionError(f"record count must equal {expected}")
    for index, record in enumerate(records):
        if int(record["record_index"]) != index:
            raise AssertionError("record index mismatch")
        reconstruct_record(record, config)
    summary, analysis = analyze(records)
    after = {
        path.name: _sha256(path)
        for path in (config_path, records_path, environment_path)
    }
    if before != after:
        raise AssertionError("strict analyzer modified frozen input artifacts")
    analysis["frozen_input_hashes"] = before
    if args.write_results:
        _write_json_atomic(result_dir / "summary.json", summary)
        _write_json_atomic(result_dir / "analysis.json", analysis)
        with (result_dir / "checks.log").open("a", encoding="utf-8") as handle:
            handle.write(
                f"PASS strict analyzer reconstructed {len(records)} records; "
                f"classification={summary['classification']}\n"
            )
    print(
        f"PASS FP-KERN-001 strict analysis with {len(records)} records; "
        f"classification={summary['classification']}"
    )


if __name__ == "__main__":
    main()
