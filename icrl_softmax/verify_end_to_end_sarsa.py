"""公式级核验：从原始 transition 字段到 Q-memory 写回的完整 softmax SARSA 构造。"""

from __future__ import annotations

import json
import math

import torch

from model import (
    EndToEndFiniteSoftmaxSARSA,
    EndToEndMaskedSoftmaxSARSA,
    GroupedSoftmaxMax,
)


DTYPE = torch.float64


def exact_batch_reference(
    q_values, states, actions, rewards, next_states, next_actions, gamma, alpha
):
    n_actions = q_values.shape[1]
    current_pairs = states * n_actions + actions
    next_pairs = next_states * n_actions + next_actions
    q_flat = q_values.reshape(-1)
    td_values = rewards + gamma * q_flat[next_pairs] - q_flat[current_pairs]

    updated = q_flat.clone()
    unique_pairs = torch.unique(current_pairs, sorted=True)
    for pair in unique_pairs:
        matching = current_pairs == pair
        updated[pair] += alpha * td_values[matching].mean()
    return updated.reshape_as(q_values), td_values, current_pairs, next_pairs


def fixed_fixture():
    q_values = torch.tensor(
        [
            [0.20, -0.35, 0.80],
            [1.10, -0.60, 0.05],
            [0.45, 0.30, -0.90],
            [0.70, -0.15, 0.55],
        ],
        dtype=DTYPE,
    )
    current_pairs = torch.tensor([0, 0, 1, 4, 4, 4, 7], dtype=torch.long)
    next_pairs = torch.tensor([4, 2, 7, 8, 0, 11, 3], dtype=torch.long)
    states = current_pairs // q_values.shape[1]
    actions = current_pairs % q_values.shape[1]
    next_states = next_pairs // q_values.shape[1]
    next_actions = next_pairs % q_values.shape[1]
    rewards = torch.tensor([0.4, -0.2, 0.7, -0.8, 0.1, 0.9, -0.5], dtype=DTYPE)
    return q_values, states, actions, rewards, next_states, next_actions


def make_literal_layout(n_pairs):
    """Coordinate ledger for the column-token matrix witness."""
    d_model = 2 * n_pairs + 8
    return {
        "m": n_pairs,
        "d": d_model,
        "q_type": 0,
        "transition_type": 1,
        "null_type": 2,
        "current_ids": slice(3, 3 + n_pairs),
        "next_ids": slice(3 + n_pairs, 3 + 2 * n_pairs),
        "r": 2 * n_pairs + 3,
        "q": 2 * n_pairs + 4,
        "u": 2 * n_pairs + 5,
        "v": 2 * n_pairs + 6,
        "delta": 2 * n_pairs + 7,
    }


def build_literal_prompt(
    q_values, states, actions, rewards, next_states, next_actions
):
    """Build H^(0)=[Q-memory | transitions | null] with tokens as columns."""
    EndToEndMaskedSoftmaxSARSA._validate_inputs(
        q_values, states, actions, rewards, next_states, next_actions
    )
    n_states, n_actions = q_values.shape
    device = q_values.device
    dtype = q_values.dtype
    states = states.to(device=device, dtype=torch.long)
    actions = actions.to(device=device, dtype=torch.long)
    next_states = next_states.to(device=device, dtype=torch.long)
    next_actions = next_actions.to(device=device, dtype=torch.long)
    rewards = rewards.to(device=device, dtype=dtype)

    current_pairs = states * n_actions + actions
    next_pairs = next_states * n_actions + next_actions
    n_pairs = n_states * n_actions
    batch_size = len(states)
    layout = make_literal_layout(n_pairs)
    n_tokens = n_pairs + batch_size + 1

    prompt = torch.zeros(
        (layout["d"], n_tokens), dtype=dtype, device=device
    )
    pair_basis = torch.eye(n_pairs, dtype=dtype, device=device)
    transition_columns = slice(n_pairs, n_pairs + batch_size)

    prompt[layout["q_type"], :n_pairs] = 1.0
    prompt[layout["current_ids"], :n_pairs] = pair_basis
    prompt[layout["q"], :n_pairs] = q_values.reshape(-1)

    prompt[layout["transition_type"], transition_columns] = 1.0
    prompt[layout["current_ids"], transition_columns] = pair_basis[
        current_pairs
    ].transpose(0, 1)
    prompt[layout["next_ids"], transition_columns] = pair_basis[
        next_pairs
    ].transpose(0, 1)
    prompt[layout["r"], transition_columns] = rewards

    prompt[layout["null_type"], -1] = 1.0
    return prompt, layout, current_pairs, next_pairs


def make_id_selector(layout, block, dtype, device):
    """Return an out-by-in matrix selecting one of the m-dimensional ID blocks."""
    selector = torch.zeros(
        (layout["m"], layout["d"]), dtype=dtype, device=device
    )
    selector[:, layout[block]] = torch.eye(
        layout["m"], dtype=dtype, device=device
    )
    return selector


def make_scalar_selector(layout, coordinate, dtype, device):
    """Return the 1-by-d row selector for a named scalar coordinate."""
    selector = torch.zeros((1, layout["d"]), dtype=dtype, device=device)
    selector[0, layout[coordinate]] = 1.0
    return selector


def make_retrieval_allowed(layout, pair_queries, batch_size):
    """Allow transition queries their unique memory source and all others null."""
    n_pairs = layout["m"]
    n_tokens = n_pairs + batch_size + 1
    null_column = n_tokens - 1
    allowed = torch.zeros(
        (n_tokens, n_tokens), dtype=torch.bool, device=pair_queries.device
    )
    transition_rows = n_pairs + torch.arange(
        batch_size, device=pair_queries.device
    )
    allowed[transition_rows, pair_queries] = True
    allowed[:n_pairs, null_column] = True
    allowed[null_column, null_column] = True
    return allowed


def make_write_allowed(layout, current_pairs, batch_size):
    """Build every write-back mask row, including all null-token branches."""
    n_pairs = layout["m"]
    n_tokens = n_pairs + batch_size + 1
    null_column = n_tokens - 1
    pair_axis = torch.arange(n_pairs, device=current_pairs.device)
    match = current_pairs[:, None] == pair_axis[None, :]
    visited = match.any(dim=0)

    allowed = torch.zeros(
        (n_tokens, n_tokens), dtype=torch.bool, device=current_pairs.device
    )
    allowed[:n_pairs, n_pairs : n_pairs + batch_size] = match.transpose(0, 1)
    unvisited_rows = pair_axis[~visited]
    allowed[unvisited_rows, null_column] = True
    allowed[n_pairs:, null_column] = True
    return allowed, visited


def literal_attention(hidden, w_q, w_k, w_v, w_o, allowed):
    """Literal column-token softmax head with query-by-source attention rows."""
    n_tokens = hidden.shape[1]
    if allowed.shape != (n_tokens, n_tokens):
        raise AssertionError("attention mask must have shape (query, source)")
    if not torch.all(allowed.any(dim=-1)):
        raise AssertionError("every attention query must admit at least one source")
    if w_q.shape[1] != hidden.shape[0] or w_k.shape != w_q.shape:
        raise AssertionError("query/key projections have inconsistent shapes")
    if w_v.shape[1] != hidden.shape[0] or w_o.shape[1] != w_v.shape[0]:
        raise AssertionError("value/output projections have inconsistent shapes")
    if w_o.shape[0] != hidden.shape[0]:
        raise AssertionError("head output must return to the residual width")

    queries = w_q @ hidden
    keys = w_k @ hidden
    values = w_v @ hidden
    raw_logits = queries.transpose(0, 1) @ keys / math.sqrt(w_q.shape[0])
    masked_logits = raw_logits.masked_fill(~allowed, float("-inf"))
    attention = torch.softmax(masked_logits, dim=-1)
    if not torch.all(torch.isfinite(attention)):
        raise AssertionError("masked softmax produced a non-finite weight")
    output = w_o @ (values @ attention.transpose(0, 1))
    return output, attention, masked_logits


def run_literal_matrix_sarsa(
    q_values,
    states,
    actions,
    rewards,
    next_states,
    next_actions,
    gamma,
    alpha,
):
    """Execute the two-block matrix construction declared in the proof."""
    hidden_0, layout, current_pairs, next_pairs = build_literal_prompt(
        q_values, states, actions, rewards, next_states, next_actions
    )
    dtype = q_values.dtype
    device = q_values.device
    n_pairs = layout["m"]
    batch_size = len(states)

    select_current = make_id_selector(
        layout, "current_ids", dtype=dtype, device=device
    )
    select_next = make_id_selector(layout, "next_ids", dtype=dtype, device=device)
    select_q = make_scalar_selector(layout, "q", dtype=dtype, device=device)
    select_r = make_scalar_selector(layout, "r", dtype=dtype, device=device)
    select_u = make_scalar_selector(layout, "u", dtype=dtype, device=device)
    select_v = make_scalar_selector(layout, "v", dtype=dtype, device=device)
    select_delta = make_scalar_selector(
        layout, "delta", dtype=dtype, device=device
    )

    current_allowed = make_retrieval_allowed(layout, current_pairs, batch_size)
    next_allowed = make_retrieval_allowed(layout, next_pairs, batch_size)
    current_output, current_attention, current_logits = literal_attention(
        hidden_0,
        select_current,
        select_current,
        select_q,
        select_u.transpose(0, 1),
        current_allowed,
    )
    next_output, next_attention, next_logits = literal_attention(
        hidden_0,
        select_next,
        select_current,
        select_q,
        select_v.transpose(0, 1),
        next_allowed,
    )
    hidden_retrieved = hidden_0 + current_output + next_output

    g = select_r + float(gamma) * select_v - select_u
    w_1 = torch.cat((g, -g), dim=0)
    w_2 = torch.zeros((layout["d"], 2), dtype=dtype, device=device)
    w_2[layout["delta"], :] = torch.tensor(
        [1.0, -1.0], dtype=dtype, device=device
    )
    ffn_preactivation = w_1 @ hidden_retrieved
    ffn_output = w_2 @ torch.relu(ffn_preactivation)
    hidden_residual = hidden_retrieved + ffn_output

    write_allowed, visited = make_write_allowed(layout, current_pairs, batch_size)
    write_output, write_attention, write_logits = literal_attention(
        hidden_residual,
        select_current,
        select_current,
        select_delta,
        float(alpha) * select_q.transpose(0, 1),
        write_allowed,
    )
    hidden_final = hidden_residual + write_output
    q_new = hidden_final[layout["q"], :n_pairs].reshape_as(q_values)

    diagnostics = {
        "layout": layout,
        "current_pairs": current_pairs,
        "next_pairs": next_pairs,
        "hidden_0": hidden_0,
        "hidden_retrieved": hidden_retrieved,
        "hidden_residual": hidden_residual,
        "hidden_final": hidden_final,
        "current_allowed": current_allowed,
        "next_allowed": next_allowed,
        "write_allowed": write_allowed,
        "current_attention": current_attention,
        "next_attention": next_attention,
        "write_attention": write_attention,
        "current_logits": current_logits,
        "next_logits": next_logits,
        "write_logits": write_logits,
        "ffn_preactivation": ffn_preactivation,
        "ffn_output": ffn_output,
        "visited": visited,
        "projections": {
            "select_current": select_current,
            "select_next": select_next,
            "select_q": select_q,
            "select_r": select_r,
            "select_u": select_u,
            "select_v": select_v,
            "select_delta": select_delta,
            "current_w_q": select_current,
            "current_w_k": select_current,
            "current_w_v": select_q,
            "current_w_o": select_u.transpose(0, 1),
            "next_w_q": select_next,
            "next_w_k": select_current,
            "next_w_v": select_q,
            "next_w_o": select_v.transpose(0, 1),
            "ffn_w_1": w_1,
            "ffn_w_2": w_2,
            "write_w_q": select_current,
            "write_w_k": select_current,
            "write_w_v": select_delta,
            "write_w_o": float(alpha) * select_q.transpose(0, 1),
        },
    }
    return q_new, diagnostics


def check_literal_matrix_construction():
    gamma, alpha = 0.61, 0.37
    inputs = fixed_fixture()
    q_values, states, actions, rewards, next_states, next_actions = inputs
    literal, diagnostics = run_literal_matrix_sarsa(
        *inputs, gamma=gamma, alpha=alpha
    )
    compact, compact_diagnostics = EndToEndMaskedSoftmaxSARSA(
        gamma=gamma, alpha=alpha
    )(*inputs)
    expected, td_values, current_pairs, next_pairs = exact_batch_reference(
        *inputs, gamma, alpha
    )

    layout = diagnostics["layout"]
    n_pairs = q_values.numel()
    batch_size = len(states)
    n_tokens = n_pairs + batch_size + 1
    transition_columns = slice(n_pairs, n_pairs + batch_size)
    hidden_0 = diagnostics["hidden_0"]
    hidden_retrieved = diagnostics["hidden_retrieved"]
    hidden_residual = diagnostics["hidden_residual"]
    hidden_final = diagnostics["hidden_final"]
    q_flat = q_values.reshape(-1)

    assert layout["d"] == 2 * n_pairs + 8
    assert hidden_0.shape == (layout["d"], n_tokens)
    assert torch.equal(
        hidden_0[layout["current_ids"], :n_pairs],
        torch.eye(n_pairs, dtype=DTYPE),
    )
    assert torch.equal(hidden_0[layout["q"], :n_pairs], q_flat)
    assert torch.all(hidden_0[layout["q_type"], :n_pairs] == 1.0)
    assert torch.all(
        hidden_0[layout["transition_type"], transition_columns] == 1.0
    )
    assert hidden_0[layout["null_type"], -1].item() == 1.0
    assert torch.count_nonzero(
        hidden_0[
            [layout["q"], layout["u"], layout["v"], layout["delta"]],
            transition_columns,
        ]
    ).item() == 0
    assert torch.count_nonzero(
        hidden_0[
            [layout["r"], layout["q"], layout["u"], layout["v"], layout["delta"]],
            -1,
        ]
    ).item() == 0

    projections = diagnostics["projections"]
    assert projections["current_w_q"].shape == (n_pairs, layout["d"])
    assert projections["current_w_k"].shape == (n_pairs, layout["d"])
    assert projections["current_w_v"].shape == (1, layout["d"])
    assert projections["current_w_o"].shape == (layout["d"], 1)
    assert projections["next_w_q"].shape == (n_pairs, layout["d"])
    assert projections["next_w_k"].shape == (n_pairs, layout["d"])
    assert projections["next_w_v"].shape == (1, layout["d"])
    assert projections["next_w_o"].shape == (layout["d"], 1)
    assert projections["ffn_w_1"].shape == (2, layout["d"])
    assert projections["ffn_w_2"].shape == (layout["d"], 2)
    assert projections["write_w_q"].shape == (n_pairs, layout["d"])
    assert projections["write_w_k"].shape == (n_pairs, layout["d"])
    assert projections["write_w_v"].shape == (1, layout["d"])
    assert projections["write_w_o"].shape == (layout["d"], 1)

    for prefix in ("current", "next", "write"):
        allowed = diagnostics[f"{prefix}_allowed"]
        attention = diagnostics[f"{prefix}_attention"]
        assert allowed.shape == (n_tokens, n_tokens)
        assert attention.shape == (n_tokens, n_tokens)
        assert torch.all(allowed.any(dim=-1))
        assert torch.all(torch.isfinite(attention))
        assert torch.allclose(
            attention.sum(dim=-1),
            torch.ones(n_tokens, dtype=DTYPE),
            atol=1e-14,
        )
        assert torch.count_nonzero(attention.masked_select(~allowed)).item() == 0

    null_column = n_tokens - 1
    for prefix, pairs in (("current", current_pairs), ("next", next_pairs)):
        allowed = diagnostics[f"{prefix}_allowed"]
        for k, pair in enumerate(pairs.tolist()):
            row = n_pairs + k
            assert int(allowed[row].sum()) == 1
            assert allowed[row, pair]
        assert torch.all(allowed[:n_pairs, null_column])
        assert torch.all(allowed[:n_pairs].sum(dim=-1) == 1)
        assert allowed[null_column, null_column]
        assert int(allowed[null_column].sum()) == 1

    write_allowed = diagnostics["write_allowed"]
    visited = diagnostics["visited"]
    for pair in range(n_pairs):
        matching = current_pairs == pair
        if matching.any():
            assert torch.equal(write_allowed[pair, transition_columns], matching)
            assert not write_allowed[pair, null_column]
            assert int(write_allowed[pair].sum()) == int(matching.sum())
        else:
            assert int(write_allowed[pair].sum()) == 1
            assert write_allowed[pair, null_column]
    assert torch.all(write_allowed[n_pairs:, null_column])
    assert torch.all(write_allowed[n_pairs:].sum(dim=-1) == 1)

    assert torch.equal(hidden_retrieved[:, :n_pairs], hidden_0[:, :n_pairs])
    assert torch.equal(hidden_retrieved[:, -1], hidden_0[:, -1])
    assert torch.allclose(
        hidden_retrieved[layout["u"], transition_columns],
        q_flat[current_pairs],
        atol=1e-14,
    )
    assert torch.allclose(
        hidden_retrieved[layout["v"], transition_columns],
        q_flat[next_pairs],
        atol=1e-14,
    )
    preserved_rows = torch.ones(layout["d"], dtype=torch.bool)
    preserved_rows[[layout["u"], layout["v"]]] = False
    assert torch.equal(
        hidden_retrieved[preserved_rows, transition_columns],
        hidden_0[preserved_rows, transition_columns],
    )

    assert (td_values > 0).any() and (td_values < 0).any()
    assert diagnostics["ffn_preactivation"].shape == (2, n_tokens)
    assert torch.allclose(
        diagnostics["ffn_preactivation"][0, transition_columns],
        td_values,
        atol=1e-14,
    )
    assert torch.allclose(
        diagnostics["ffn_preactivation"][1, transition_columns],
        -td_values,
        atol=1e-14,
    )
    assert torch.allclose(
        hidden_residual[layout["delta"], transition_columns],
        td_values,
        atol=1e-14,
    )
    assert torch.count_nonzero(
        hidden_residual[layout["delta"], :n_pairs]
    ).item() == 0
    assert hidden_residual[layout["delta"], -1].item() == 0.0
    non_delta_rows = torch.ones(layout["d"], dtype=torch.bool)
    non_delta_rows[layout["delta"]] = False
    assert torch.equal(
        hidden_residual[non_delta_rows], hidden_retrieved[non_delta_rows]
    )

    assert torch.allclose(literal, expected, atol=1e-14)
    assert torch.allclose(literal, compact, atol=1e-14)
    assert torch.equal(literal.reshape(-1)[~visited], q_flat[~visited])
    assert torch.equal(hidden_final[:, n_pairs:], hidden_residual[:, n_pairs:])
    final_preserved_rows = torch.ones(layout["d"], dtype=torch.bool)
    final_preserved_rows[layout["q"]] = False
    assert torch.equal(
        hidden_final[final_preserved_rows], hidden_residual[final_preserved_rows]
    )

    assert torch.equal(
        diagnostics["current_attention"][transition_columns, :n_pairs],
        compact_diagnostics["current_attention"],
    )
    assert torch.equal(
        diagnostics["next_attention"][transition_columns, :n_pairs],
        compact_diagnostics["next_attention"],
    )
    assert torch.equal(
        diagnostics["write_attention"][:n_pairs, n_pairs:].transpose(0, 1),
        compact_diagnostics["write_attention"],
    )

    write_attention = diagnostics["write_attention"]
    for pair in range(n_pairs):
        matching = current_pairs == pair
        if matching.any():
            weights = write_attention[pair, transition_columns][matching]
            assert torch.allclose(
                weights,
                torch.full_like(weights, 1.0 / int(matching.sum())),
                atol=1e-14,
            )
        else:
            assert write_attention[pair, null_column].item() == 1.0

    return {
        "prompt_shape": list(hidden_0.shape),
        "literal_reference_max_abs_error": torch.max(
            torch.abs(literal - expected)
        ).item(),
        "literal_compact_max_abs_error": torch.max(
            torch.abs(literal - compact)
        ).item(),
        "minimum_mask_support": min(
            int(diagnostics[f"{prefix}_allowed"].sum(dim=-1).min())
            for prefix in ("current", "next", "write")
        ),
    }


def check_exact_masked_construction():
    gamma, alpha = 0.61, 0.37
    inputs = fixed_fixture()
    q_values, states, actions, rewards, next_states, next_actions = inputs
    model = EndToEndMaskedSoftmaxSARSA(gamma=gamma, alpha=alpha)
    actual, diagnostics = model(*inputs)
    expected, td_values, current_pairs, next_pairs = exact_batch_reference(
        *inputs, gamma, alpha
    )

    q_flat = q_values.reshape(-1)
    retrieval_error = max(
        torch.max(torch.abs(diagnostics["current_q"] - q_flat[current_pairs])).item(),
        torch.max(torch.abs(diagnostics["next_q"] - q_flat[next_pairs])).item(),
    )
    residual_error = torch.max(
        torch.abs(diagnostics["td_values"] - td_values)
    ).item()
    update_error = torch.max(torch.abs(actual - expected)).item()

    assert retrieval_error <= 1e-14
    assert residual_error <= 1e-14
    assert update_error <= 1e-14
    assert torch.allclose(
        diagnostics["current_attention"].sum(dim=-1),
        torch.ones(len(states), dtype=DTYPE),
        atol=1e-14,
    )
    assert torch.allclose(
        diagnostics["next_attention"].sum(dim=-1),
        torch.ones(len(states), dtype=DTYPE),
        atol=1e-14,
    )
    assert torch.allclose(
        diagnostics["write_attention"].sum(dim=0),
        torch.ones(q_values.numel(), dtype=DTYPE),
        atol=1e-14,
    )

    visited = diagnostics["visited"]
    actual_flat = actual.reshape(-1)
    assert torch.equal(actual_flat[~visited], q_flat[~visited])

    # Matching transition weights are exactly 1/n_x; unvisited queries place
    # unit mass on the final null-token row.
    write_attention = diagnostics["write_attention"]
    for pair in range(q_values.numel()):
        matching = current_pairs == pair
        if matching.any():
            expected_weight = 1.0 / int(matching.sum())
            assert torch.allclose(
                write_attention[:-1, pair][matching],
                torch.full((int(matching.sum()),), expected_weight, dtype=DTYPE),
                atol=1e-14,
            )
            assert write_attention[-1, pair].item() == 0.0
        else:
            assert write_attention[-1, pair].item() == 1.0

    return {
        "retrieval_max_abs_error": retrieval_error,
        "residual_max_abs_error": residual_error,
        "update_max_abs_error": update_error,
        "unvisited_pairs": int((~visited).sum()),
    }


def check_sequential_equivalence():
    gamma, alpha = 0.53, 0.19
    q_attention = torch.tensor(
        [[0.2, -0.1], [0.7, 0.0], [-0.3, 0.5]], dtype=DTYPE
    )
    q_reference = q_attention.clone()
    q_literal = q_attention.clone()
    current_pairs = [0, 3, 3, 4, 1, 5]
    next_pairs = [3, 4, 0, 1, 5, 2]
    rewards = [0.6, -0.2, 0.1, 0.8, -0.5, 0.3]
    model = EndToEndMaskedSoftmaxSARSA(gamma=gamma, alpha=alpha)

    for current_pair, next_pair, reward in zip(current_pairs, next_pairs, rewards):
        n_actions = q_attention.shape[1]
        fields = (
            torch.tensor([current_pair // n_actions], dtype=torch.long),
            torch.tensor([current_pair % n_actions], dtype=torch.long),
            torch.tensor([reward], dtype=DTYPE),
            torch.tensor([next_pair // n_actions], dtype=torch.long),
            torch.tensor([next_pair % n_actions], dtype=torch.long),
        )
        q_attention, _ = model(q_attention, *fields)
        q_literal, _ = run_literal_matrix_sarsa(
            q_literal, *fields, gamma=gamma, alpha=alpha
        )

        flat = q_reference.reshape(-1)
        td_value = reward + gamma * flat[next_pair].item() - flat[current_pair].item()
        flat[current_pair] += alpha * td_value
        q_reference = flat.reshape_as(q_reference)
        assert torch.allclose(q_attention, q_reference, atol=1e-14)
        assert torch.allclose(q_literal, q_reference, atol=1e-14)

    return {
        "iterate_max_abs_error": torch.max(
            torch.abs(q_attention - q_reference)
        ).item(),
        "literal_iterate_max_abs_error": torch.max(
            torch.abs(q_literal - q_reference)
        ).item(),
    }


def check_finite_logit_bound():
    gamma, alpha = 0.61, 0.37
    retrieval_beta, kernel_beta = 2.4, 2.1
    inputs = fixed_fixture()
    q_values, states, actions, rewards, next_states, next_actions = inputs
    finite = EndToEndFiniteSoftmaxSARSA(
        gamma=gamma,
        alpha=alpha,
        retrieval_beta=retrieval_beta,
        kernel_beta=kernel_beta,
    )
    actual, diagnostics = finite(*inputs)
    exact, true_td, current_pairs, _ = exact_batch_reference(
        *inputs, gamma, alpha
    )

    n_pairs = q_values.numel()
    batch_size = len(states)
    q_span = (torch.max(q_values) - torch.min(q_values)).item()
    retrieval_leakage = (n_pairs - 1) / (math.exp(retrieval_beta) + n_pairs - 1)
    residual_bound = (1.0 + gamma) * retrieval_leakage * q_span
    td_bound = torch.max(torch.abs(true_td)).item()

    per_query = []
    for pair in diagnostics["query_pairs"].tolist():
        count = int((current_pairs == pair).sum())
        write_l1 = 2.0 * (batch_size - count) / (
            count * math.exp(kernel_beta) + batch_size - count
        )
        bound = alpha * (residual_bound + td_bound * write_l1)
        error = abs(actual.reshape(-1)[pair].item() - exact.reshape(-1)[pair].item())
        assert error <= bound + 1e-12
        per_query.append((pair, error, bound))

    true_current = q_values.reshape(-1)[current_pairs]
    retrieval_error = torch.max(
        torch.abs(diagnostics["current_q"] - true_current)
    ).item()
    assert retrieval_error <= retrieval_leakage * q_span + 1e-12

    return {
        "retrieval_max_abs_error": retrieval_error,
        "retrieval_error_bound": retrieval_leakage * q_span,
        "update_max_abs_error": max(row[1] for row in per_query),
        "update_max_bound": max(row[2] for row in per_query),
    }


def check_internal_boltzmann_head():
    q_actions = torch.tensor(
        [[0.2, -0.4, 1.1], [-0.7, 0.5, 0.3], [1.0, 1.0, -0.2]],
        dtype=DTYPE,
    )
    beta = 3.7
    value, attention = GroupedSoftmaxMax(beta)(q_actions)
    explicit_attention = torch.softmax(beta * q_actions, dim=-1)
    explicit_value = (explicit_attention * q_actions).sum(dim=-1)
    assert torch.allclose(attention, explicit_attention, atol=1e-14)
    assert torch.allclose(value, explicit_value, atol=1e-14)
    return {
        "probability_max_abs_error": torch.max(
            torch.abs(attention - explicit_attention)
        ).item(),
        "expectation_max_abs_error": torch.max(torch.abs(value - explicit_value)).item(),
    }


def main():
    results = {
        "literal_matrix": check_literal_matrix_construction(),
        "exact_masked": check_exact_masked_construction(),
        "sequential": check_sequential_equivalence(),
        "finite_logit": check_finite_logit_bound(),
        "boltzmann": check_internal_boltzmann_head(),
    }
    print(json.dumps(results, indent=2))
    print("PASS end-to-end softmax SARSA construction")


if __name__ == "__main__":
    main()
