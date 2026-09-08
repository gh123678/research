"""Strict regression and acceptance analysis for FP-ADV-001 outputs.

Verifies the frozen FP-TU-001 baseline hashes, replays every serialized
action-gap gate from observable quantities, rechecks the dominance contracts
and the policy-update algebra, and (for the 480-record formal protocol)
regresses all legacy leaves against the frozen FP-TU-001 results.  Smoke runs
are regressed on the matched baseline subset selected by
(seed_entropy, spawn_key, trajectory_length, task_index).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import action_gap_certificate as agc
from evaluate_visit_indexed_certificates import write_json


TASK_ID = "FP-ADV-001"
PROBABILITY_STATEMENT = (
    "P(EmitUpdate and (any used action ordering is false "
    "or exists s: V^{pi_plus}(s) < V^pi(s))) <= delta"
)
FPTU_BASELINE_DIR = Path(
    r"C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-TU-001\codex"
)
BASELINE_HASHES = {
    "config.json": "43dcb96b0f6f95e76f1c0b484d6375e3727dbb5609b16a8952a69e2ac0dddf3a",
    "task_results.json": "0e5eab39bf49894832c5ebcdd6f7b70c453fff6b9600b234617889f8f9fa79be",
    "summary.json": "565fc4d261a938d13350984bb97242e79fba42f014d11f807a18942517f4444f",
}
NAMESPACE = "action_gap_certificate"
TOL = 1e-12
LOCAL_ONLY_REASONS = {"attention_mass_invalid", "effective_transition_row_invalid"}
ROUTE_SPECS = {
    "vfirst_local_exact": ("vfirst", "exact", "local", "exact", None),
    "vfirst_local_softmax": ("vfirst", "softmax", "local", "softmax", None),
    "vfirst_global_exact": ("vfirst", "exact", "global", "exact", "vfirst_nosplit_exact"),
    "vfirst_global_softmax": ("vfirst", "softmax", "global", "softmax", "vfirst_nosplit_softmax"),
    "direct_global_exact": ("direct", "exact", "global", None, "direct_exact"),
    "direct_global_softmax": ("direct", "softmax", "global", None, "direct_softmax"),
}
ROUTE_NAMES = tuple(ROUTE_SPECS)


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


def _close(expected: float, observed: float) -> bool:
    if not math.isfinite(expected) or not math.isfinite(observed):
        return False
    return math.isclose(expected, observed, rel_tol=TOL, abs_tol=TOL)


def _compare(expected: Any, observed: Any, path: str, mismatches: list[str]) -> None:
    if isinstance(expected, bool) or isinstance(observed, bool):
        if expected is not observed:
            mismatches.append(path)
        return
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if not _close(float(expected), float(observed)):
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


def _without_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_namespace(item)
            for key, item in value.items()
            if key != NAMESPACE
        }
    if isinstance(value, list):
        return [_without_namespace(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(r"C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-ADV-001\claude"),
    )
    parser.add_argument("--baseline-dir", type=Path, default=FPTU_BASELINE_DIR)
    return parser.parse_args()


def _replay_donor_reason(
    *,
    family: str,
    matching: str,
    scope: str,
    mode_ok: bool,
    diverged: bool,
    state_value_bound: float | None,
    global_q_bound: float | None,
    receiver_count: int,
    donor_count: int,
    radius_a: float | None,
    radius_b: float | None,
) -> str | None:
    """Replay the ordered gates that depend only on serialized scalars."""
    if not mode_ok:
        return "algorithm_mode_mismatch"
    if diverged:
        return "divergence_guard_triggered"
    if family == "vfirst" and state_value_bound is None:
        return "state_certificate_not_emitted"
    if receiver_count == 0 or donor_count == 0:
        return "candidate_pair_unvisited"
    if scope == "local" and (radius_a is None or radius_b is None):
        return "recovery_radius_unavailable"
    if scope == "global" and global_q_bound is None:
        return "recovery_radius_unavailable"
    return None


def audit_route_record(
    *,
    record: dict[str, Any],
    route_name: str,
    route: dict[str, Any],
    config: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    family, matching, scope, state_key, eq_route = ROUTE_SPECS[route_name]
    where = f"record {record['task_index']}:{record['spawn_key']}:{route_name}"
    tu = record["time_uniform_certificate"]
    algorithm = tu["certificate_inputs"]["algorithm"]
    mode_ok = bool(
        algorithm["mode"] == "fixed_policy_synchronous"
        and algorithm["fixed_context"]
        and algorithm["synchronous_update"]
    )
    diverged = (
        bool(algorithm[f"{eq_route}_diverged"]) if family == "direct" else False
    )
    gamma = float(config["gamma"])
    beta = float(record["beta"])
    length = int(record["trajectory_length"])
    n_actions = int(record["n_actions"])
    pi_min = float(record["pi_min"])
    reward_bound = 1.0 + float(record["gap_bonus"])
    value_bound = reward_bound / (1.0 - gamma)
    if not _close(value_bound, float(route["declared_value_bound"])):
        raise AssertionError(f"{where}: declared value bound mismatch")
    if route["family"] != family or route["matching"] != matching or route["scope"] != scope:
        raise AssertionError(f"{where}: route labels mismatch")

    pair_counts = tu["certificate_inputs"]["observed"]["pair_counts"]
    radii = tu["event"]["recovery"]["radius_by_group"]
    state_bound_expected = (
        tu["state_value"][state_key]["total_bound"] if family == "vfirst" else None
    )
    if state_bound_expected is None:
        if route["state_value_bound"] is not None:
            raise AssertionError(f"{where}: unexpected state bound")
    elif not _close(float(state_bound_expected), float(route["state_value_bound"])):
        raise AssertionError(f"{where}: state value bound mismatch")
    eq_expected = (
        tu["routes"][eq_route]["total_bound"] if scope == "global" else None
    )
    if eq_expected is None:
        if route["global_q_bound"] is not None:
            raise AssertionError(f"{where}: unexpected global Q bound")
    elif not _close(float(eq_expected), float(route["global_q_bound"])):
        raise AssertionError(f"{where}: global Q bound mismatch")

    changed_expected: list[int] = []
    reason_pool: list[str] = []
    for state_entry in route["states"]:
        state = int(state_entry["state"])
        q_row = [float(value) for value in state_entry["q_hat_row"]]
        if agc.select_receiver(q_row) != int(state_entry["receiver"]):
            raise AssertionError(f"{where}: receiver selection mismatch in state {state}")
        receiver = int(state_entry["receiver"])
        g_receiver = state * n_actions + receiver
        receiver_count = int(pair_counts[g_receiver])
        if receiver_count != int(state_entry["receiver_count"]):
            raise AssertionError(f"{where}: receiver count mismatch in state {state}")
        base_row = [0.0] * n_actions
        base_row[receiver] = float(state_entry["receiver_pi_before"])
        transferred = 0.0
        for donor in state_entry["donors"]:
            action = int(donor["action"])
            g_donor = state * n_actions + action
            donor_count = int(pair_counts[g_donor])
            if donor_count != int(donor["pair_count"]):
                raise AssertionError(f"{where}: donor count mismatch s={state} a={action}")
            base_row[action] = float(donor["pi_before"])
            reason = donor["reason"]
            if reason is not None:
                if reason not in agc.REASON_RANK:
                    raise AssertionError(f"{where}: unknown reason {reason}")
                reason_pool.append(str(reason))
            replayed = _replay_donor_reason(
                family=family,
                matching=matching,
                scope=scope,
                mode_ok=mode_ok,
                diverged=diverged,
                state_value_bound=route["state_value_bound"],
                global_q_bound=route["global_q_bound"],
                receiver_count=receiver_count,
                donor_count=donor_count,
                radius_a=radii[g_receiver] if scope == "local" else None,
                radius_b=radii[g_donor] if scope == "local" else None,
            )
            if replayed is not None:
                if reason != replayed:
                    raise AssertionError(
                        f"{where}: gate mismatch s={state} a={action}: "
                        f"{reason} != {replayed}"
                    )
                for key in ("kappa", "total_variation", "uncertainty", "lcb"):
                    if donor[key] is not None:
                        raise AssertionError(f"{where}: blocked donor carries {key}")
                if donor["eligible"] or float(donor["transfer"]) != 0.0:
                    raise AssertionError(f"{where}: blocked donor transfers mass")
                continue
            if reason in LOCAL_ONLY_REASONS:
                if scope != "local" or matching != "softmax":
                    raise AssertionError(f"{where}: {reason} outside softmax local")
                if donor["uncertainty"] is not None or donor["lcb"] is not None:
                    raise AssertionError(f"{where}: invalid-row donor carries bounds")
                if donor["eligible"] or float(donor["transfer"]) != 0.0:
                    raise AssertionError(f"{where}: invalid-row donor transfers mass")
                if donor["kappa"] is not None:
                    kappa_expected = agc.softmax_on_group_mass(donor_count, length, beta)
                    if not _close(kappa_expected, float(donor["kappa"])):
                        raise AssertionError(f"{where}: kappa mismatch s={state} a={action}")
                stats["invalid_row_reasons"][str(reason)] += 1
                continue
            if scope == "global":
                if donor["total_variation"] is not None:
                    raise AssertionError(f"{where}: global donor carries total variation")
                tv = None
            else:
                if donor["total_variation"] is None:
                    raise AssertionError(f"{where}: evaluated donor lacks total variation")
                tv = float(donor["total_variation"])
                if not (0.0 <= tv <= 1.0):
                    raise AssertionError(f"{where}: total variation outside [0,1]")
            if matching == "softmax" and scope == "local":
                kappa_expected = agc.softmax_on_group_mass(donor_count, length, beta)
                if donor["kappa"] is None or not _close(kappa_expected, float(donor["kappa"])):
                    raise AssertionError(f"{where}: kappa mismatch s={state} a={action}")
            elif donor["kappa"] is not None:
                raise AssertionError(f"{where}: unexpected kappa on {matching} {scope}")
            if scope == "global":
                uncertainty = agc.global_uncertainty(float(route["global_q_bound"]))
            elif matching == "exact":
                uncertainty = agc.exact_local_uncertainty(
                    float(radii[g_receiver]),
                    float(radii[g_donor]),
                    gamma,
                    float(route["state_value_bound"]),
                    tv,
                )
            else:
                kappa_a = agc.softmax_on_group_mass(receiver_count, length, beta)
                kappa_b = agc.softmax_on_group_mass(donor_count, length, beta)
                contamination_a = agc.softmax_contamination(
                    kappa_a, float(radii[g_receiver]), value_bound
                )
                contamination_b = agc.softmax_contamination(
                    kappa_b, float(radii[g_donor]), value_bound
                )
                uncertainty = agc.softmax_local_uncertainty(
                    contamination_a,
                    contamination_b,
                    gamma,
                    float(route["state_value_bound"]),
                    tv,
                )
            difference = q_row[receiver] - q_row[action]
            lcb = difference - uncertainty
            if not math.isfinite(lcb) or not math.isfinite(uncertainty):
                if reason != "numerical_nonfinite":
                    raise AssertionError(f"{where}: nonfinite result without guard")
                stats["nonfinite_guard"] += 1
                continue
            if not _close(uncertainty, float(donor["uncertainty"])):
                raise AssertionError(f"{where}: uncertainty mismatch s={state} a={action}")
            if not _close(lcb, float(donor["lcb"])):
                raise AssertionError(f"{where}: lcb mismatch s={state} a={action}")
            if lcb <= 0.0:
                expected_reason: str | None = "gap_lcb_nonpositive"
            elif float(donor["pi_before"]) <= pi_min:
                expected_reason = "no_transferable_mass"
            else:
                expected_reason = None
            if reason != expected_reason:
                raise AssertionError(
                    f"{where}: tail gate mismatch s={state} a={action}: "
                    f"{reason} != {expected_reason}"
                )
            if expected_reason is None:
                if not donor["eligible"]:
                    raise AssertionError(f"{where}: eligible flag mismatch")
                transfer = 0.5 * (float(donor["pi_before"]) - pi_min)
                if float(donor["transfer"]) != transfer:
                    raise AssertionError(f"{where}: transfer mismatch s={state} a={action}")
                transferred += transfer
            else:
                if donor["eligible"] or float(donor["transfer"]) != 0.0:
                    raise AssertionError(f"{where}: abstaining donor transfers mass")
        if float(state_entry["transferred_mass"]) != transferred:
            raise AssertionError(f"{where}: transferred mass mismatch in state {state}")
        if bool(state_entry["update_applied"]) != (transferred > 0.0):
            raise AssertionError(f"{where}: update flag mismatch in state {state}")
        if transferred > 0.0:
            changed_expected.append(state)
        stats["transfers_by_route"][route_name].append(transferred)

    if route["changed_states"] != changed_expected:
        raise AssertionError(f"{where}: changed states mismatch")
    if int(route["changed_state_count"]) != len(changed_expected):
        raise AssertionError(f"{where}: changed state count mismatch")
    if int(route["eligible_donor_count"]) != sum(
        1
        for state_entry in route["states"]
        for donor in state_entry["donors"]
        if donor["eligible"]
    ):
        raise AssertionError(f"{where}: eligible donor count mismatch")
    total_mass = sum(float(state_entry["transferred_mass"]) for state_entry in route["states"])
    if not _close(total_mass, float(route["total_transferred_mass"])):
        raise AssertionError(f"{where}: total transferred mass mismatch")
    emitted = int(route["eligible_donor_count"]) > 0
    if bool(route["update_emitted"]) != emitted:
        raise AssertionError(f"{where}: emission flag mismatch")
    expected_reasons = sorted(
        set(reason_pool),
        key=lambda reason: (agc.REASON_RANK.get(reason, len(agc.REASON_ORDER)), reason),
    )
    if route["reasons"] != expected_reasons:
        raise AssertionError(f"{where}: ordered reasons mismatch")
    policy_plus = route["policy_plus"]
    if not emitted:
        if policy_plus is not None:
            raise AssertionError(f"{where}: abstention carries a policy")
        return
    for state_entry in route["states"]:
        state = int(state_entry["state"])
        receiver = int(state_entry["receiver"])
        expected_row = [0.0] * n_actions
        expected_row[receiver] = float(state_entry["receiver_pi_before"])
        for donor in state_entry["donors"]:
            expected_row[int(donor["action"])] = float(donor["pi_before"])
        if state_entry["update_applied"]:
            for donor in state_entry["donors"]:
                if donor["eligible"]:
                    expected_row[int(donor["action"])] -= float(donor["transfer"])
            expected_row[receiver] += float(state_entry["transferred_mass"])
        serialized_row = policy_plus[state]
        for action in range(n_actions):
            if not _close(expected_row[action], float(serialized_row[action])):
                raise AssertionError(
                    f"{where}: policy_plus mismatch s={state} a={action}"
                )
        if abs(sum(float(v) for v in serialized_row) - 1.0) > 1e-9:
            raise AssertionError(f"{where}: policy_plus row {state} not normalized")
        if min(float(v) for v in serialized_row) < pi_min - 1e-12:
            raise AssertionError(f"{where}: policy_plus row {state} breaks the floor")


def audit_record(
    record: dict[str, Any], config: dict[str, Any], stats: dict[str, Any]
) -> None:
    stripped = _without_namespace(record)
    if set(record) - set(stripped) != {NAMESPACE}:
        raise AssertionError("record has an invalid additive namespace")
    namespace = record[NAMESPACE]
    if namespace["task_id"] != TASK_ID:
        raise AssertionError("namespace task id mismatch")
    if namespace["probability_statement"] != PROBABILITY_STATEMENT:
        raise AssertionError("probability statement mismatch")
    if not _close(float(namespace["delta"]), float(config["certificate_delta"])):
        raise AssertionError("namespace delta mismatch")
    if float(namespace["transfer_fraction"]) != 0.5:
        raise AssertionError("transfer fraction is not the frozen 0.5")
    agc.assert_no_oracle_keys(
        {key: value for key, value in namespace.items() if key != "oracle_audit"}
    )
    for route_name in ROUTE_NAMES:
        audit_route_record(
            record=record,
            route_name=route_name,
            route=namespace["routes"][route_name],
            config=config,
            stats=stats,
        )
        stats["emissions"][route_name][
            int(record["trajectory_length"])
        ] += int(namespace["routes"][route_name]["update_emitted"])
        for reason in namespace["routes"][route_name]["reasons"]:
            stats["route_reasons"][route_name][reason] += 1
    dominance = namespace["dominance"]
    for label, local_name, global_name in (
        ("exact", "vfirst_local_exact", "vfirst_global_exact"),
        ("softmax", "vfirst_local_softmax", "vfirst_global_softmax"),
    ):
        expected = agc.check_local_global_dominance(
            namespace["routes"][local_name], namespace["routes"][global_name]
        )
        observed = dominance[label]
        if bool(expected["available"]) != bool(observed["available"]):
            raise AssertionError(f"dominance availability mismatch for {label}")
        if expected["available"]:
            if expected["satisfied"] is not True or observed["satisfied"] is not True:
                raise AssertionError(f"local/global dominance failed for {label}")
            if int(expected["checked_pairs"]) != int(observed["checked_pairs"]):
                raise AssertionError(f"dominance pair count mismatch for {label}")
            stats["dominance_checked"][label] += int(expected["checked_pairs"])
        weak_expected = agc.check_weak_update_dominance(
            namespace["routes"][local_name], namespace["routes"][global_name]
        )
        weak_observed = dominance[f"weak_update_{label}"]
        if bool(weak_expected["available"]) != bool(weak_observed["available"]):
            raise AssertionError(f"weak dominance availability mismatch for {label}")
        if weak_expected["available"] and weak_expected["satisfied"] is not True:
            raise AssertionError(f"weak update dominance failed for {label}")
    audit = namespace["oracle_audit"]
    stats["false_ordering"] += int(audit["any_false_ordering"])
    stats["bound_violation"] += int(audit["any_bound_violation"])
    stats["bellman_violation"] += int(audit["any_bellman_violation"])
    stats["value_decrease"] += int(audit["any_value_decrease"])
    for route_name in ROUTE_NAMES:
        route_audit = audit["routes"][route_name]
        if route_audit["update_emitted"]:
            stats["emitted_records"][route_name] += 1
            stats["receiver_true_greedy"][route_name] += int(
                route_audit["receiver_true_greedy_states"]
            )
            stats["state_totals"][route_name] += int(route_audit["state_count"])
            bellman = route_audit["min_bellman_improvement"]
            value = route_audit["min_value_improvement"]
            if bellman is None or value is None:
                raise AssertionError(f"emitted update lacks Bellman audit for {route_name}")
            stats["min_bellman"][route_name] = min(
                stats["min_bellman"][route_name], float(bellman)
            )
            stats["min_value"][route_name] = min(
                stats["min_value"][route_name], float(value)
            )
            stats["return_change"][route_name].append(float(route_audit["return_change"]))
        stats["false_ordering_donors"][route_name] += int(
            route_audit["false_ordering_count"]
        )
        stats["bound_violation_donors"][route_name] += int(
            route_audit["bound_violation_count"]
        )


def make_stats() -> dict[str, Any]:
    return {
        "emissions": {route: Counter() for route in ROUTE_NAMES},
        "route_reasons": {route: Counter() for route in ROUTE_NAMES},
        "transfers_by_route": {route: [] for route in ROUTE_NAMES},
        "invalid_row_reasons": Counter(),
        "nonfinite_guard": 0,
        "dominance_checked": Counter(),
        "emitted_records": Counter(),
        "receiver_true_greedy": Counter(),
        "state_totals": Counter(),
        "min_bellman": defaultdict(lambda: math.inf),
        "min_value": defaultdict(lambda: math.inf),
        "return_change": {route: [] for route in ROUTE_NAMES},
        "false_ordering": 0,
        "bound_violation": 0,
        "bellman_violation": 0,
        "value_decrease": 0,
        "false_ordering_donors": Counter(),
        "bound_violation_donors": Counter(),
    }


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(record["seed_entropy"]),
        tuple(int(item) for item in record["spawn_key"]),
        int(record["trajectory_length"]),
        int(record["task_index"]),
        int(record["n_actions"]),
        float(record["pi_min"]),
        float(record["beta"]),
        float(record["mixing"]),
        float(record["gap_bonus"]),
    )


def audit_summary(
    summary: list[dict[str, Any]], records: list[dict[str, Any]]
) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for legacy_route, new_routes in SUMMARY_ROUTE_MAP.items():
            key = (
                int(record["trajectory_length"]),
                int(record["n_actions"]),
                float(record["pi_min"]),
                float(record["beta"]),
                float(record["mixing"]),
                float(record["gap_bonus"]),
                legacy_route,
            )
            grouped[key].append(record)
    for row in summary:
        namespace = row.get(NAMESPACE)
        if namespace is None:
            continue
        key = (
            int(row["trajectory_length"]),
            int(row["n_actions"]),
            float(row["pi_min"]),
            float(row["beta"]),
            float(row["mixing"]),
            float(row["gap_bonus"]),
            str(row["route"]),
        )
        selected = grouped.get(key)
        if selected is None:
            raise AssertionError(f"summary row {key} has no matching records")
        for route_name, route_stats in namespace["routes"].items():
            if route_name not in ROUTE_SPECS:
                raise AssertionError(f"summary carries unknown route {route_name}")
            emitted = sum(
                int(item[NAMESPACE]["routes"][route_name]["update_emitted"])
                for item in selected
            )
            if int(route_stats["update_emission_count"]) != emitted:
                raise AssertionError(f"summary emission mismatch for {route_name}")
            if int(route_stats["records"]) != len(selected):
                raise AssertionError(f"summary record count mismatch for {route_name}")


SUMMARY_ROUTE_MAP = {
    "direct_exact": ("direct_global_exact",),
    "direct_softmax": ("direct_global_softmax",),
    "vfirst_nosplit_exact": ("vfirst_local_exact", "vfirst_global_exact"),
    "vfirst_nosplit_softmax": ("vfirst_local_softmax", "vfirst_global_softmax"),
}


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    baseline_dir = args.baseline_dir.resolve()
    observed_hashes = {name: sha256_file(baseline_dir / name) for name in BASELINE_HASHES}
    if observed_hashes != BASELINE_HASHES:
        raise RuntimeError(f"baseline hash mismatch: {observed_hashes}")
    baseline_records = load_strict_json(baseline_dir / "task_results.json")
    config = load_strict_json(result_dir / "config.json")
    records = load_strict_json(result_dir / "task_results.json")
    summary = load_strict_json(result_dir / "summary.json")
    formal = len(records) == 480
    mismatches: list[str] = []
    if formal:
        baseline_config = load_strict_json(baseline_dir / "config.json")
        baseline_summary = load_strict_json(baseline_dir / "summary.json")
        _compare(baseline_config, _without_namespace(config), "config", mismatches)
        _compare(baseline_records, _without_namespace(records), "records", mismatches)
        _compare(baseline_summary, _without_namespace(summary), "summary", mismatches)
        if mismatches:
            raise AssertionError(
                f"legacy regression has {len(mismatches)} mismatches: {mismatches[:10]}"
            )
    else:
        if not records:
            raise AssertionError("smoke result is empty")
        baseline_index = {record_key(record): record for record in baseline_records}
        matched = 0
        for index, record in enumerate(records):
            baseline_record = baseline_index.get(record_key(record))
            if baseline_record is None:
                raise AssertionError(f"smoke record {index} has no baseline match")
            _compare(
                baseline_record,
                _without_namespace(record),
                f"records[{index}]",
                mismatches,
            )
            matched += 1
        if mismatches:
            raise AssertionError(
                f"smoke regression has {len(mismatches)} mismatches: {mismatches[:10]}"
            )
    stats = make_stats()
    for record in records:
        audit_record(record, config, stats)
    audit_summary(summary, records)
    emissions = {
        route: dict(sorted(counter.items()))
        for route, counter in stats["emissions"].items()
    }
    record_audit = {
        "record_count": len(records),
        "formal_protocol": formal,
        "update_emissions_by_route_and_length": emissions,
        "route_reasons": {
            route: dict(sorted(counter.items()))
            for route, counter in stats["route_reasons"].items()
        },
        "dominance_checked_pairs": dict(sorted(stats["dominance_checked"].items())),
        "invalid_row_reasons": dict(sorted(stats["invalid_row_reasons"].items())),
        "nonfinite_guard_count": stats["nonfinite_guard"],
        "oracle_audit": {
            "used_as_theorem_evidence": False,
            "records_with_false_ordering": stats["false_ordering"],
            "records_with_bound_violation": stats["bound_violation"],
            "records_with_bellman_violation": stats["bellman_violation"],
            "records_with_value_decrease": stats["value_decrease"],
            "false_ordering_donors_by_route": dict(
                sorted(stats["false_ordering_donors"].items())
            ),
            "bound_violation_donors_by_route": dict(
                sorted(stats["bound_violation_donors"].items())
            ),
            "emitted_records_by_route": dict(sorted(stats["emitted_records"].items())),
            "receiver_true_greedy_rate_by_route": {
                route: (
                    stats["receiver_true_greedy"][route] / stats["state_totals"][route]
                    if stats["state_totals"][route]
                    else None
                )
                for route in ROUTE_NAMES
            },
            "min_bellman_improvement_by_route": {
                route: (
                    None
                    if math.isinf(stats["min_bellman"][route])
                    else stats["min_bellman"][route]
                )
                for route in ROUTE_NAMES
            },
            "min_value_improvement_by_route": {
                route: (
                    None
                    if math.isinf(stats["min_value"][route])
                    else stats["min_value"][route]
                )
                for route in ROUTE_NAMES
            },
            "mean_return_change_by_route": {
                route: (
                    sum(stats["return_change"][route]) / len(stats["return_change"][route])
                    if stats["return_change"][route]
                    else None
                )
                for route in ROUTE_NAMES
            },
        },
    }
    regression = {
        "status": "PASS",
        "baseline_hashes": observed_hashes,
        "baseline_record_count": len(baseline_records),
        "result_record_count": len(records),
        "formal_protocol": formal,
        "legacy_mismatch_count": len(mismatches),
        "numeric_tolerance": {"relative": TOL, "absolute": TOL},
        "record_audit": record_audit,
    }
    write_json(result_dir / "regression.json", regression)
    write_json(result_dir / "summary.json", summary)
    checks = [
        "PASS strict JSON with duplicate and nonfinite rejection",
        f"PASS frozen FP-TU-001 baseline hashes and {len(baseline_records)} records",
        f"PASS legacy regression mismatches={len(mismatches)} formal={formal}",
        "PASS ordered-gate replay, uncertainty/LCB recomputation, and transfer algebra",
        "PASS policy row sums, exploration floor, and theta=0.5 transfer formula",
        "PASS local/global dominance and weak update dominance rechecks",
        "PASS oracle separation; "
        f"records with value decrease={stats['value_decrease']}",
    ]
    with (result_dir / "checks.log").open("a", encoding="utf-8") as handle:
        handle.write("\n" + "\n".join(checks) + "\n")
    for name in (
        "config.json",
        "task_results.json",
        "summary.json",
        "regression.json",
        "environment.json",
    ):
        load_strict_json(result_dir / name)
    hashes_after = {name: sha256_file(baseline_dir / name) for name in BASELINE_HASHES}
    if hashes_after != observed_hashes:
        raise RuntimeError("baseline changed during analysis")
    print(f"PASS {TASK_ID} analysis: records={len(records)}, formal={formal}")


if __name__ == "__main__":
    main()
