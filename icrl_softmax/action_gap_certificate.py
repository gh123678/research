"""Pure action-gap certificates and one safe policy update for FP-ADV-001.

This module consumes only observable or declared quantities: visit counts,
observed successor histograms, route Q estimates, the current policy, the
emitted FP-TU-001 state/recovery/complete-Q bounds, and public
hyperparameters.  It performs no sampling, no file writing, no plotting, and
no truth computation.  True kernels, occupancies, value or Q tables, action
gaps, realized errors, exact returns, and oracle diagonals are prohibited
inputs and are rejected by ``assert_no_oracle_keys``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from fixed_policy_finite_sample_certificate import strict_json_ready


REASON_ORDER = (
    "algorithm_mode_mismatch",
    "divergence_guard_triggered",
    "state_certificate_not_emitted",
    "candidate_pair_unvisited",
    "recovery_radius_unavailable",
    "attention_mass_invalid",
    "effective_transition_row_invalid",
    "numerical_nonfinite",
    "gap_lcb_nonpositive",
    "no_transferable_mass",
)
REASON_RANK = {reason: index for index, reason in enumerate(REASON_ORDER)}

FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "true_kernel",
        "true_transition",
        "true_transition_kernel",
        "occupancy",
        "true_occupancy",
        "true_q",
        "q_pi",
        "true_value",
        "v_pi",
        "true_v",
        "true_reward",
        "true_residual",
        "action_gap_true",
        "true_action_gap",
        "realized_error",
        "exact_return",
        "true_return",
        "oracle",
        "oracle_diagonal",
        "bellman_improvement_true",
        "value_improvement_true",
    }
)

DOMINANCE_TOLERANCE = 1e-12


class ActionGapInputError(ValueError):
    """Structurally invalid observable input to the pure certificate."""


def strict_json(value: Any) -> Any:
    return strict_json_ready(value)


def assert_no_oracle_keys(mapping: Mapping[str, Any], path: str = "") -> None:
    for key, value in mapping.items():
        name = str(key)
        if name in FORBIDDEN_INPUT_KEYS or name.startswith("oracle_"):
            raise ActionGapInputError(f"prohibited oracle input at {path}{name}")
        if isinstance(value, Mapping):
            assert_no_oracle_keys(value, path=f"{path}{name}.")


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ActionGapInputError(f"{name} must be finite")
    return number


def _nonnegative_integer(name: str, value: int) -> int:
    integer = int(value)
    if isinstance(value, bool) or integer != value or integer < 0:
        raise ActionGapInputError(f"{name} must be a nonnegative integer")
    return integer


def _canonical_reasons(reasons: Sequence[str]) -> list[str]:
    return sorted(
        set(reasons),
        key=lambda reason: (REASON_RANK.get(reason, len(REASON_ORDER)), reason),
    )


def softmax_on_group_mass(count: int, trajectory_length: int, beta: float) -> float:
    """Return the normalized one-hot softmax on-group mass kappa_g."""
    visits = _nonnegative_integer("count", count)
    length = _nonnegative_integer("trajectory_length", trajectory_length)
    sharpness = _finite("beta", beta)
    if visits <= 0 or visits > length:
        raise ActionGapInputError("count must lie in [1, trajectory_length]")
    if visits == length:
        return 1.0
    logit = sharpness + math.log(visits) - math.log(length - visits)
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _validate_count_row(name: str, row: Sequence[int]) -> list[int]:
    values = [_nonnegative_integer(f"{name}[{index}]", value) for index, value in enumerate(row)]
    if not values:
        raise ActionGapInputError(f"{name} must be nonempty")
    return values


def exact_successor_row(successor_counts_row: Sequence[int]) -> list[float]:
    """Normalize one observed successor histogram into a probability row."""
    counts = _validate_count_row("successor_counts_row", successor_counts_row)
    total = sum(counts)
    if total <= 0:
        raise ActionGapInputError("successor row has zero total count")
    return [value / total for value in counts]


def softmax_successor_row(
    successor_counts_row: Sequence[int],
    total_successor_counts: Sequence[int],
    beta: float,
) -> tuple[list[float], float]:
    """Return the normalized effective successor row and on-group mass."""
    counts = _validate_count_row("successor_counts_row", successor_counts_row)
    totals = _validate_count_row("total_successor_counts", total_successor_counts)
    if len(counts) != len(totals):
        raise ActionGapInputError("successor row dimensions differ")
    sharpness = _finite("beta", beta)
    n_g = sum(counts)
    total = sum(totals)
    if n_g <= 0 or n_g > total:
        raise ActionGapInputError("group count must lie in [1, trajectory_length]")
    for index, (mine, all_) in enumerate(zip(counts, totals)):
        if mine > all_:
            raise ActionGapInputError(
                f"group successor count exceeds total at index {index}"
            )
    kappa = softmax_on_group_mass(n_g, total, sharpness)
    denominator = n_g * math.exp(sharpness) + (total - n_g)
    row = [
        (math.exp(sharpness) * mine + (all_ - mine)) / denominator
        for mine, all_ in zip(counts, totals)
    ]
    return row, kappa


def total_variation(row_p: Sequence[float], row_q: Sequence[float]) -> float:
    if len(row_p) != len(row_q) or not row_p:
        raise ActionGapInputError("total-variation rows must share a nonempty support")
    value = 0.5 * sum(abs(float(p) - float(q)) for p, q in zip(row_p, row_q))
    if not math.isfinite(value):
        raise ActionGapInputError("total variation is nonfinite")
    return value


def softmax_contamination(kappa: float, radius: float, value_bound: float) -> float:
    """Return C_g = kappa_g r_g + 2B(1 - kappa_g)."""
    mass = _finite("kappa", kappa)
    rad = _finite("radius", radius)
    bound = _finite("value_bound", value_bound)
    if not 0.0 < mass <= 1.0:
        raise ActionGapInputError("kappa must lie in (0,1]")
    if rad < 0.0 or bound <= 0.0:
        raise ActionGapInputError("radius must be nonnegative and B positive")
    return mass * rad + 2.0 * bound * (1.0 - mass)


def exact_local_uncertainty(
    radius_a: float,
    radius_b: float,
    gamma: float,
    state_value_bound: float,
    tv: float,
) -> float:
    """Return r_sa + r_sb + 2 gamma E_V TV(P_bar_sa, P_bar_sb)."""
    return (
        _finite("radius_a", radius_a)
        + _finite("radius_b", radius_b)
        + 2.0 * _finite("gamma", gamma) * _finite("state_value_bound", state_value_bound)
        * _finite("tv", tv)
    )


def softmax_local_uncertainty(
    contamination_a: float,
    contamination_b: float,
    gamma: float,
    state_value_bound: float,
    tv: float,
) -> float:
    """Return C_sa + C_sb + 2 gamma E_V^beta TV(P_bar_sa^beta, P_bar_sb^beta)."""
    return (
        _finite("contamination_a", contamination_a)
        + _finite("contamination_b", contamination_b)
        + 2.0 * _finite("gamma", gamma) * _finite("state_value_bound", state_value_bound)
        * _finite("tv", tv)
    )


def global_uncertainty(global_q_bound: float) -> float:
    """Return the complete-Q control penalty 2 E_Q."""
    return 2.0 * _finite("global_q_bound", global_q_bound)


def select_receiver(q_row: Sequence[float]) -> int:
    """Return argmax_a q_hat(s,a) with smallest-index tie breaking."""
    values = [_finite(f"q_row[{index}]", value) for index, value in enumerate(q_row)]
    if not values:
        raise ActionGapInputError("q_row must be nonempty")
    best = 0
    for index, value in enumerate(values):
        if value > values[best]:
            best = index
    return best


def _validate_policy(policy: Sequence[Sequence[float]], pi_min: float) -> list[list[float]]:
    rows = [[_finite(f"policy[{s}][{a}]", value) for a, value in enumerate(row)] for s, row in enumerate(policy)]
    if not rows or not rows[0]:
        raise ActionGapInputError("policy must be a nonempty matrix")
    width = len(rows[0])
    for s, row in enumerate(rows):
        if len(row) != width:
            raise ActionGapInputError("policy rows must share one width")
        if any(value < 0.0 for value in row):
            raise ActionGapInputError(f"policy row {s} has a negative entry")
        if abs(sum(row) - 1.0) > 1e-9:
            raise ActionGapInputError(f"policy row {s} is not normalized")
        if any(value < pi_min - 1e-12 for value in row):
            raise ActionGapInputError(f"policy row {s} violates the exploration floor")
    return rows


def evaluate_route_update(
    *,
    family: str,
    matching: str,
    scope: str,
    q_estimate: Sequence[Sequence[float]],
    policy: Sequence[Sequence[float]],
    pi_min: float,
    theta: float,
    gamma: float,
    beta: float,
    trajectory_length: int,
    reward_bound: float,
    pair_counts: Sequence[int],
    successor_counts: Sequence[Sequence[int]] | None = None,
    total_successor_counts: Sequence[int] | None = None,
    recovery_radii: Sequence[float | None] | None = None,
    state_value_bound: float | None = None,
    global_q_bound: float | None = None,
    algorithm_mode_ok: bool = True,
    diverged: bool = False,
) -> dict[str, Any]:
    """Evaluate one route's certified action gaps and the single safe update."""
    if family not in ("vfirst", "direct"):
        raise ActionGapInputError("family must be vfirst or direct")
    if matching not in ("exact", "softmax"):
        raise ActionGapInputError("matching must be exact or softmax")
    if scope not in ("local", "global"):
        raise ActionGapInputError("scope must be local or global")
    if family == "direct" and scope == "local":
        raise ActionGapInputError("the task defines no local Direct-Q route")
    discount = _finite("gamma", gamma)
    if not 0.0 < discount < 1.0:
        raise ActionGapInputError("gamma must lie in (0,1)")
    sharpness = _finite("beta", beta)
    reward = _finite("reward_bound", reward_bound)
    if reward <= 0.0:
        raise ActionGapInputError("reward_bound must be positive")
    floor = _finite("pi_min", pi_min)
    fraction = _finite("theta", theta)
    if not 0.0 < fraction < 1.0:
        raise ActionGapInputError("theta must lie in (0,1)")
    horizon = _nonnegative_integer("trajectory_length", trajectory_length)
    if horizon <= 0:
        raise ActionGapInputError("trajectory_length must be positive")
    if not isinstance(algorithm_mode_ok, bool) or not isinstance(diverged, bool):
        raise ActionGapInputError("mode and divergence flags must be boolean")
    value_bound = reward / (1.0 - discount)

    counts = [
        _nonnegative_integer(f"pair_counts[{index}]", value)
        for index, value in enumerate(pair_counts)
    ]
    q_rows = [[_finite(f"q_estimate[{s}][{a}]", value) for a, value in enumerate(row)] for s, row in enumerate(q_estimate)]
    if not q_rows or not q_rows[0]:
        raise ActionGapInputError("q_estimate must be a nonempty matrix")
    n_states = len(q_rows)
    n_actions = len(q_rows[0])
    if any(len(row) != n_actions for row in q_rows):
        raise ActionGapInputError("q_estimate rows must share one width")
    if len(counts) != n_states * n_actions:
        raise ActionGapInputError("pair_counts length must equal n_states*n_actions")
    if sum(counts) != horizon:
        raise ActionGapInputError("pair_counts must sum to trajectory_length")
    policy_rows = _validate_policy(policy, floor)
    if len(policy_rows) != n_states or len(policy_rows[0]) != n_actions:
        raise ActionGapInputError("policy shape must match q_estimate")
    if floor <= 0.0 or floor >= 1.0 / n_actions:
        raise ActionGapInputError("pi_min must lie in (0, 1/n_actions)")

    if state_value_bound is not None:
        state_value_bound = _finite("state_value_bound", state_value_bound)
        if state_value_bound < 0.0:
            raise ActionGapInputError("state_value_bound must be nonnegative")
    if global_q_bound is not None:
        global_q_bound = _finite("global_q_bound", global_q_bound)
        if global_q_bound < 0.0:
            raise ActionGapInputError("global_q_bound must be nonnegative")

    histograms: list[list[int]] = []
    successor_totals: list[int] = []
    radii: list[float | None] = []
    if scope == "local":
        if successor_counts is None or total_successor_counts is None or recovery_radii is None:
            raise ActionGapInputError("local routes require successor counts and radii")
        if len(successor_counts) != n_states * n_actions:
            raise ActionGapInputError("successor_counts must have one row per pair")
        histograms = [_validate_count_row(f"successor_counts[{g}]", row) for g, row in enumerate(successor_counts)]
        for g, row in enumerate(histograms):
            if len(row) != n_states:
                raise ActionGapInputError("successor rows must have one entry per state")
            if sum(row) != counts[g]:
                raise ActionGapInputError(f"successor row {g} must sum to its pair count")
        successor_totals = _validate_count_row("total_successor_counts", total_successor_counts)
        if len(successor_totals) != n_states:
            raise ActionGapInputError("total_successor_counts must have one entry per state")
        if sum(successor_totals) != horizon:
            raise ActionGapInputError("total_successor_counts must sum to trajectory_length")
        for index, total in enumerate(successor_totals):
            if sum(row[index] for row in histograms) != total:
                raise ActionGapInputError("successor columns must match total_successor_counts")
        if len(recovery_radii) != n_states * n_actions:
            raise ActionGapInputError("recovery_radii must have one entry per pair")
        for index, radius in enumerate(recovery_radii):
            if radius is None:
                radii.append(None)
                continue
            value = _finite(f"recovery_radii[{index}]", radius)
            if value < 0.0:
                raise ActionGapInputError("recovery radii must be nonnegative")
            radii.append(value)

    def donor_evaluation(state: int, receiver: int, donor: int) -> dict[str, Any]:
        g_receiver = state * n_actions + receiver
        g_donor = state * n_actions + donor
        entry: dict[str, Any] = {
            "action": donor,
            "pair_count": counts[g_donor],
            "kappa": None,
            "total_variation": None,
            "uncertainty": None,
            "lcb": None,
            "pi_before": policy_rows[state][donor],
            "transfer": 0.0,
            "eligible": False,
            "reason": None,
        }
        reason: str | None = None
        uncertainty: float | None = None
        if not algorithm_mode_ok:
            reason = "algorithm_mode_mismatch"
        elif diverged:
            reason = "divergence_guard_triggered"
        elif family == "vfirst" and state_value_bound is None:
            reason = "state_certificate_not_emitted"
        elif counts[g_receiver] == 0 or counts[g_donor] == 0:
            reason = "candidate_pair_unvisited"
        if reason is None and scope == "local":
            radius_a = radii[g_receiver]
            radius_b = radii[g_donor]
            if radius_a is None or radius_b is None:
                reason = "recovery_radius_unavailable"
            else:
                try:
                    if matching == "exact":
                        row_a = exact_successor_row(histograms[g_receiver])
                        row_b = exact_successor_row(histograms[g_donor])
                        kappa_a = kappa_b = None
                    else:
                        row_a, kappa_a = softmax_successor_row(
                            histograms[g_receiver], successor_totals, sharpness
                        )
                        row_b, kappa_b = softmax_successor_row(
                            histograms[g_donor], successor_totals, sharpness
                        )
                        entry["kappa"] = kappa_b
                except ActionGapInputError:
                    row_a = row_b = kappa_a = kappa_b = None
                if matching == "softmax":
                    masses = (kappa_a, kappa_b)
                    if any(
                        mass is None
                        or not math.isfinite(float(mass))
                        or not 0.0 < float(mass) <= 1.0
                        for mass in masses
                    ):
                        reason = "attention_mass_invalid"
                if reason is None and (row_a is None or row_b is None):
                    reason = "effective_transition_row_invalid"
                if reason is None:
                    tv = total_variation(row_a, row_b)
                    entry["total_variation"] = tv
                    if matching == "exact":
                        uncertainty = exact_local_uncertainty(
                            radius_a, radius_b, discount, float(state_value_bound), tv
                        )
                    else:
                        contamination_a = softmax_contamination(
                            float(kappa_a), float(radius_a), value_bound
                        )
                        contamination_b = softmax_contamination(
                            float(kappa_b), float(radius_b), value_bound
                        )
                        uncertainty = softmax_local_uncertainty(
                            contamination_a,
                            contamination_b,
                            discount,
                            float(state_value_bound),
                            tv,
                        )
        elif reason is None and scope == "global":
            if global_q_bound is None:
                reason = "recovery_radius_unavailable"
            else:
                uncertainty = global_uncertainty(global_q_bound)
        if reason is None:
            difference = q_rows[state][receiver] - q_rows[state][donor]
            lcb = difference - float(uncertainty)
            if not math.isfinite(lcb) or not math.isfinite(float(uncertainty)):
                reason = "numerical_nonfinite"
                entry["uncertainty"] = uncertainty if math.isfinite(float(uncertainty)) else None
            else:
                entry["uncertainty"] = float(uncertainty)
                entry["lcb"] = lcb
                if lcb <= 0.0:
                    reason = "gap_lcb_nonpositive"
                elif policy_rows[state][donor] <= floor:
                    reason = "no_transferable_mass"
        if reason is None:
            entry["eligible"] = True
            entry["transfer"] = fraction * (policy_rows[state][donor] - floor)
        entry["reason"] = reason
        return entry

    states_out: list[dict[str, Any]] = []
    policy_plus = [list(row) for row in policy_rows]
    changed_states: list[int] = []
    eligible_donor_count = 0
    total_transferred = 0.0
    for state in range(n_states):
        receiver = select_receiver(q_rows[state])
        donors = [
            donor_evaluation(state, receiver, donor)
            for donor in range(n_actions)
            if donor != receiver
        ]
        transferred = sum(donor["transfer"] for donor in donors if donor["eligible"])
        if transferred > 0.0:
            for donor in donors:
                if donor["eligible"]:
                    policy_plus[state][donor["action"]] -= donor["transfer"]
            policy_plus[state][receiver] += transferred
            changed_states.append(state)
            eligible_donor_count += sum(1 for donor in donors if donor["eligible"])
            total_transferred += transferred
        states_out.append(
            {
                "state": state,
                "receiver": receiver,
                "receiver_count": counts[state * n_actions + receiver],
                "receiver_pi_before": policy_rows[state][receiver],
                "q_hat_row": list(q_rows[state]),
                "donors": donors,
                "transferred_mass": transferred,
                "update_applied": transferred > 0.0,
            }
        )

    emitted = eligible_donor_count > 0
    reasons = _canonical_reasons(
        [
            donor["reason"]
            for state_entry in states_out
            for donor in state_entry["donors"]
            if donor["reason"] is not None
        ]
    )
    return {
        "family": family,
        "matching": matching,
        "scope": scope,
        "state_value_bound": state_value_bound if family == "vfirst" else None,
        "global_q_bound": global_q_bound if scope == "global" else None,
        "declared_value_bound": value_bound,
        "update_emitted": emitted,
        "changed_states": changed_states,
        "changed_state_count": len(changed_states),
        "eligible_donor_count": eligible_donor_count,
        "total_transferred_mass": total_transferred,
        "policy_plus": policy_plus if emitted else None,
        "reasons": reasons,
        "states": states_out,
    }


def check_local_global_dominance(
    local_record: dict[str, Any], global_record: dict[str, Any]
) -> dict[str, Any]:
    """Check that every local penalty is no larger than 2 E_Q when emitted."""
    bound = global_record.get("global_q_bound")
    if bound is None:
        return {
            "available": False,
            "satisfied": None,
            "checked_pairs": 0,
            "max_ratio": None,
        }
    penalty = global_uncertainty(float(bound))
    checked = 0
    max_ratio: float | None = None
    satisfied = True
    for state_entry in local_record["states"]:
        for donor in state_entry["donors"]:
            uncertainty = donor["uncertainty"]
            if uncertainty is None:
                continue
            checked += 1
            ratio = float(uncertainty) / penalty if penalty > 0.0 else math.inf
            max_ratio = ratio if max_ratio is None else max(max_ratio, ratio)
            if float(uncertainty) > penalty + DOMINANCE_TOLERANCE:
                satisfied = False
    return {
        "available": True,
        "satisfied": satisfied,
        "checked_pairs": checked,
        "max_ratio": max_ratio,
    }


def check_weak_update_dominance(
    local_record: dict[str, Any], global_record: dict[str, Any]
) -> dict[str, Any]:
    """Check record-level weak dominance of local over global update decisions."""
    if not global_record["update_emitted"]:
        return {
            "available": False,
            "satisfied": None,
            "local_transferred_mass": local_record["total_transferred_mass"],
            "global_transferred_mass": global_record["total_transferred_mass"],
        }
    satisfied = bool(
        local_record["update_emitted"]
        and set(global_record["changed_states"]).issubset(local_record["changed_states"])
        and local_record["total_transferred_mass"]
        >= global_record["total_transferred_mass"] - DOMINANCE_TOLERANCE
    )
    return {
        "available": True,
        "satisfied": satisfied,
        "local_transferred_mass": local_record["total_transferred_mass"],
        "global_transferred_mass": global_record["total_transferred_mass"],
    }
