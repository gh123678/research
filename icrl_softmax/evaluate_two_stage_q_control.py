"""统一表格 MDP 实验：从 softmax TD evaluation 到 kernelized Q control。

主实验使用期望 Bellman target，避免把有限样本噪声与 softmax-max、kernel
leakage 混在一起。每个状态动作对对应一个 context token，第二阶段仍通过真实
softmax kernel 聚合 signed TD values。
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mdps import sample_mdp
from model import GroupedSoftmaxMax, KernelizedSoftmaxQTD


def bellman_target(mdp, q, gamma, beta=None):
    """精确期望 Bellman target；beta=None 使用 oracle max。"""
    p = torch.as_tensor(mdp["P"], dtype=q.dtype)
    r = torch.as_tensor(mdp["R"], dtype=q.dtype)
    if beta is None:
        next_value = torch.amax(q, dim=-1)
    else:
        next_value, _ = GroupedSoftmaxMax(beta)(q)
    return (p * (r + gamma * next_value[None, None, :])).sum(dim=-1)


def solve_q_star(mdp, gamma, tol=1e-12, max_steps=20000):
    q = torch.zeros(mdp["nS"], mdp["nA"], dtype=torch.float64)
    for _ in range(max_steps):
        q_new = bellman_target(mdp, q, gamma, beta=None)
        if float(torch.max(torch.abs(q_new - q))) < tol:
            return q_new
        q = q_new
    raise RuntimeError("Q* value iteration did not converge")


def exact_policy_return(mdp, q, gamma):
    """Expected discounted return of the deterministic greedy policy from p0."""
    p = np.asarray(mdp["P"], dtype=np.float64)
    r = np.asarray(mdp["R"], dtype=np.float64)
    p0 = np.asarray(mdp["p0"], dtype=np.float64)
    greedy = torch.argmax(q, dim=-1).detach().cpu().numpy()
    rows = np.arange(mdp["nS"])
    p_pi = p[rows, greedy]
    r_pi = (p_pi * r[rows, greedy]).sum(axis=-1)
    value = np.linalg.solve(np.eye(mdp["nS"]) - gamma * p_pi, r_pi)
    return float(p0 @ value)


def metrics(mdp, q, q_star, gamma):
    exact_target = bellman_target(mdp, q, gamma, beta=None)
    q_error = float(torch.max(torch.abs(q - q_star)))
    residual = float(torch.max(torch.abs(exact_target - q)))
    greedy = torch.argmax(q, dim=-1)
    greedy_star = torch.argmax(q_star, dim=-1)
    policy_agreement = float((greedy == greedy_star).double().mean())
    policy_return = exact_policy_return(mdp, q, gamma)
    optimal_return = exact_policy_return(mdp, q_star, gamma)
    return {
        "q_inf_error": q_error,
        "bellman_residual": residual,
        "policy_agreement": policy_agreement,
        "policy_return": policy_return,
        "optimal_return": optimal_return,
        "policy_return_gap": max(0.0, optimal_return - policy_return),
    }


def run_operator(mdp, q_star, gamma, alpha, steps, method, beta, kernel_beta):
    n_states, n_actions = mdp["nS"], mdp["nA"]
    n_queries = n_states * n_actions
    q = torch.zeros(n_states, n_actions, dtype=torch.float64)
    eye = torch.eye(n_queries, dtype=q.dtype)
    kernel = KernelizedSoftmaxQTD(alpha=alpha, kernel_beta=kernel_beta)
    history = []
    initial_policy_return = exact_policy_return(mdp, q, gamma)

    for step in range(steps):
        if method == "exact_q_iteration":
            target = bellman_target(mdp, q, gamma, beta=None)
            q = (1.0 - alpha) * q + alpha * target
        elif method in {"oracle_kernel", "internal_softmax"}:
            target_beta = None if method == "oracle_kernel" else beta
            target = bellman_target(mdp, q, gamma, beta=target_beta)
            td_values = (target - q).reshape(-1)
            q_flat, _, _ = kernel(q.reshape(-1), td_values, eye, eye)
            q = q_flat.reshape_as(q)
        elif method == "pure_convex":
            # 受限消融：删除 signed TD value，只保留正的凸 kernel 读出。
            positive_values = torch.ones(n_queries, dtype=q.dtype)
            q_flat, _, _ = kernel(q.reshape(-1), positive_values, eye, eye)
            q = q_flat.reshape_as(q)
        else:
            raise ValueError(f"unknown method: {method}")
        if step in {0, 1, 2, 4, 9, 19, 49, 99, steps - 1}:
            history.append({"step": step + 1, **metrics(mdp, q, q_star, gamma)})

    final = metrics(mdp, q, q_star, gamma)
    final["initial_policy_return"] = initial_policy_return
    final["policy_return_gain"] = final["policy_return"] - initial_policy_return
    if method == "internal_softmax":
        m_beta, _ = GroupedSoftmaxMax(beta)(q)
        final["max_approx_error"] = float(
            torch.max(torch.amax(q, dim=-1) - m_beta)
        )
        final["max_error_bound"] = math.log(n_actions) / beta
    else:
        final["max_approx_error"] = 0.0
        final["max_error_bound"] = 0.0
    diag_mass = math.exp(kernel_beta) / (math.exp(kernel_beta) + n_queries - 1)
    final["kernel_leakage"] = 1.0 - diag_mass
    return final, history


def aggregate(rows):
    grouped = {}
    numeric = [
        "q_inf_error",
        "bellman_residual",
        "policy_agreement",
        "policy_return",
        "optimal_return",
        "policy_return_gap",
        "initial_policy_return",
        "policy_return_gain",
        "max_approx_error",
        "max_error_bound",
        "kernel_leakage",
    ]
    for row in rows:
        key = row["label"]
        grouped.setdefault(key, []).append(row)
    summary = []
    for label, group in grouped.items():
        item = {
            "label": label,
            "method": group[0]["method"],
            "beta": group[0]["beta"],
            "kernel_beta": group[0]["kernel_beta"],
            "n_tasks": len(group),
        }
        for name in numeric:
            values = np.asarray([g[name] for g in group], dtype=np.float64)
            item[f"{name}_mean"] = float(values.mean())
            item[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def make_figure(summary, outdir):
    internal = sorted(
        [x for x in summary if x["method"] == "internal_softmax"],
        key=lambda x: x["beta"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    betas = np.asarray([x["beta"] for x in internal])
    policy_gains = np.asarray([x["policy_return_gain_mean"] for x in internal])
    policy_gain_std = np.asarray([x["policy_return_gain_std"] for x in internal])
    max_errors = np.asarray([x["max_approx_error_mean"] for x in internal])
    max_bounds = np.asarray([x["max_error_bound_mean"] for x in internal])

    axes[0].errorbar(
        betas,
        policy_gains,
        yerr=policy_gain_std,
        marker="o",
        capsize=3,
        label="two-stage softmax",
    )
    oracle = next(x for x in summary if x["method"] == "oracle_kernel")
    axes[0].axhline(
        oracle["policy_return_gain_mean"],
        color="black",
        linestyle="--",
        label="oracle max",
    )
    axes[0].axhline(0.0, color="0.6", linewidth=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"action sharpness $\beta$")
    axes[0].set_ylabel(r"policy-return gain $J(\pi_T)-J(\pi_0)$")
    axes[0].legend(frameon=False)

    axes[1].plot(betas, max_errors, marker="o", label="observed max gap")
    axes[1].plot(betas, max_bounds, linestyle="--", label=r"$\log|A|/\beta$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"action sharpness $\beta$")
    axes[1].set_ylabel("softmax-max error")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "two_stage_temperature.pdf", bbox_inches="tight")
    fig.savefig(outdir / "two_stage_temperature.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_scaling_sweeps(args):
    rows = []
    sweep_tasks = min(args.tasks, args.sweep_tasks)
    reference_beta = max(args.betas)

    for task in range(sweep_tasks):
        for kernel_beta in args.kernel_betas:
            rng = np.random.default_rng(20261823 + task)
            mdp = sample_mdp(args.nS, args.nA, args.gamma, rng)
            q_star = solve_q_star(mdp, args.gamma)
            final, _ = run_operator(
                mdp,
                q_star,
                args.gamma,
                args.alpha,
                args.sweep_steps,
                "internal_softmax",
                reference_beta,
                kernel_beta,
            )
            rows.append(
                {
                    "task": task,
                    "method": "kernel_sweep",
                    "label": f"kernel beta={kernel_beta:g}",
                    "beta": reference_beta,
                    "kernel_beta": kernel_beta,
                    **final,
                }
            )

        for n_actions in args.action_counts:
            rng = np.random.default_rng(20262823 + task + 1000 * n_actions)
            mdp = sample_mdp(args.nS, n_actions, args.gamma, rng)
            q_star = solve_q_star(mdp, args.gamma)
            final, _ = run_operator(
                mdp,
                q_star,
                args.gamma,
                args.alpha,
                args.sweep_steps,
                "internal_softmax",
                reference_beta,
                args.kernel_beta,
            )
            rows.append(
                {
                    "task": task,
                    "method": "action_sweep",
                    "label": f"actions={n_actions}",
                    "beta": reference_beta,
                    "kernel_beta": args.kernel_beta,
                    "n_actions": n_actions,
                    **final,
                }
            )
    return rows, aggregate(rows)


def make_scaling_figure(summary, outdir):
    kernel = sorted(
        [x for x in summary if x["method"] == "kernel_sweep"],
        key=lambda x: x["kernel_beta"],
    )
    actions = sorted(
        [x for x in summary if x["method"] == "action_sweep"],
        key=lambda x: int(x["label"].split("=")[1]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    axes[0].plot(
        [x["kernel_leakage_mean"] for x in kernel],
        [max(x["policy_return_gap_mean"], 1e-15) for x in kernel],
        marker="o",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("kernel leakage")
    axes[0].set_ylabel("greedy-policy return gap")

    n_actions = np.asarray([int(x["label"].split("=")[1]) for x in actions])
    axes[1].errorbar(
        n_actions,
        [x["policy_return_gap_mean"] for x in actions],
        yerr=[x["policy_return_gap_std"] for x in actions],
        marker="o",
        capsize=3,
        label="policy return gap",
    )
    axes[1].plot(
        n_actions,
        [x["max_error_bound_mean"] for x in actions],
        linestyle="--",
        label=r"$\log|A|/\beta$",
    )
    axes[1].set_xlabel("number of actions")
    axes[1].set_ylabel("error / bound")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "two_stage_scaling.pdf", bbox_inches="tight")
    fig.savefig(outdir / "two_stage_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--nS", type=int, default=9)
    parser.add_argument("--nA", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--kernel-beta", type=float, default=20.0)
    parser.add_argument("--betas", type=float, nargs="+", default=[1, 2, 5, 10, 20])
    parser.add_argument("--kernel-betas", type=float, nargs="+", default=[1, 2, 5, 10, 20])
    parser.add_argument("--action-counts", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--sweep-tasks", type=int, default=10)
    parser.add_argument("--sweep-steps", type=int, default=20)
    parser.add_argument(
        "--outdir", type=Path, default=Path("results/two_stage_q_control")
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows, histories = [], {}
    methods = [
        ("exact_q_iteration", None, "exact Q iteration"),
        ("oracle_kernel", None, "oracle-max kernel Q-TD"),
    ] + [("internal_softmax", b, f"internal softmax beta={b:g}") for b in args.betas]

    for task in range(args.tasks):
        rng = np.random.default_rng(20260823 + task)
        mdp = sample_mdp(args.nS, args.nA, args.gamma, rng)
        q_star = solve_q_star(mdp, args.gamma)
        for method, beta, label in methods:
            final, history = run_operator(
                mdp,
                q_star,
                args.gamma,
                args.alpha,
                args.steps,
                method,
                beta,
                args.kernel_beta,
            )
            row = {
                "task": task,
                "method": method,
                "label": label,
                "beta": beta,
                "kernel_beta": args.kernel_beta,
                **final,
            }
            rows.append(row)
            histories[f"task{task}:{label}"] = history

    summary = aggregate(rows)
    with open(args.outdir / "raw_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(args.outdir / "histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)
    with open(args.outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(args.outdir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    make_figure(summary, args.outdir)
    sweep_rows, sweep_summary = run_scaling_sweeps(args)
    with open(args.outdir / "scaling_raw.json", "w", encoding="utf-8") as f:
        json.dump(sweep_rows, f, indent=2)
    with open(args.outdir / "scaling_summary.json", "w", encoding="utf-8") as f:
        json.dump(sweep_summary, f, indent=2)
    with open(args.outdir / "scaling_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_summary[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_summary)
    make_scaling_figure(sweep_summary, args.outdir)

    print("label | Q-inf mean±std | residual | policy return gain | return gap | max gap/bound")
    for item in summary:
        print(
            f"{item['label']:36s} | "
            f"{item['q_inf_error_mean']:.6g}±{item['q_inf_error_std']:.2g} | "
            f"{item['bellman_residual_mean']:.6g} | "
            f"{item['policy_return_gain_mean']:.4g} | "
            f"{item['policy_return_gap_mean']:.4g} | "
            f"{item['max_approx_error_mean']:.4g}/{item['max_error_bound_mean']:.4g}"
        )
    print("\nscaling sweeps")
    for item in sweep_summary:
        print(
            f"{item['label']:20s} | return-gap={item['policy_return_gap_mean']:.6g} | "
            f"leakage={item['kernel_leakage_mean']:.4g} | "
            f"bound={item['max_error_bound_mean']:.4g}"
        )
    print(f"saved to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
