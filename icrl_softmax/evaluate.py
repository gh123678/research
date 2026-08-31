"""闭环策略改进评估（03 论文 Figure 3 风格）。

在未见过的 MDP 上闭环运行 in-context 更新：
- 每步把当前窗口喂给 Transformer，用其输出的 w（或直接 teacher 更新）作为新行为策略
- 每隔 eval_every 步冻结并 MC 估计折扣 return
对比：transformer / teacher / oracle(value iteration) / random

用法：
    python evaluate.py --ckpt out_sanity/model.pt --outdir out_sanity/eval
"""
import argparse
import json
import os

import numpy as np
import torch

from mdps import sample_mdp, sample_features, eps_greedy_probs, rollout
from model import LinearAttnICRL, SoftmaxAttnICRL, SoftmaxSARSA
from train import build_prompt, sarsa_teacher


def value_iteration(mdp):
    """精确 value iteration 求 Q*，返回 Q* (nS,nA) 与 v* (nS)。"""
    P, R, gamma = mdp["P"], mdp["R"], mdp["gamma"]
    nS, nA = mdp["nS"], mdp["nA"]
    v = np.zeros(nS)
    for _ in range(2000):
        Q = np.zeros((nS, nA))
        for s in range(nS):
            for a in range(nA):
                Q[s, a] = np.sum(P[s, a] * (R[s, a] + gamma * v))
        v_new = Q.max(axis=1)
        if np.max(np.abs(v_new - v)) < 1e-9:
            v = v_new
            break
        v = v_new
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            Q[s, a] = np.sum(P[s, a] * (R[s, a] + gamma * v))
    return Q, v


def greedy_probs_from_Q(Q):
    nS, nA = Q.shape
    probs = np.zeros((nS, nA), dtype=np.float32)
    probs[np.arange(nS), np.argmax(Q, axis=1)] = 1.0
    return probs


def mc_return(mdp, probs, n_trajs, len_traj, rng):
    P, R, gamma, p0 = mdp["P"], mdp["R"], mdp["gamma"], mdp["p0"]
    nS, nA = mdp["nS"], mdp["nA"]
    Gs = np.zeros(n_trajs)
    for j in range(n_trajs):
        s = rng.choice(nS, p=p0)
        G, g = 0.0, 1.0
        for _ in range(len_traj):
            a = rng.choice(nA, p=probs[s])
            s_next = rng.choice(nS, p=P[s, a])
            G += g * R[s, a, s_next]
            g *= gamma
            s = s_next
        Gs[j] = G
    return float(Gs.mean())


def run_closed_loop(mdps, phis, use_teacher, model, args, rng, device):
    """对每个 MDP 闭环更新，返回 (n_mdps, n_evals) 的 return 矩阵。"""
    n_evals = args.update_steps // args.eval_every
    rets = np.zeros((len(mdps), n_evals))
    for m, (mdp, phi_np) in enumerate(zip(mdps, phis)):
        phi = torch.tensor(phi_np, device=device)
        w = torch.tensor(rng.uniform(-1, 1, size=args.d).astype(np.float32), device=device)
        s = int(rng.choice(args.nS, p=mdp["p0"]))
        step, ei = 0, 0
        while step < args.update_steps:
            probs = eps_greedy_probs(phi.cpu().numpy(), w.detach().cpu().numpy(), args.eps)
            S, A, Rew = rollout(mdp, probs, s, args.n, rng)
            S = torch.tensor(S, device=device)
            A = torch.tensor(A, device=device)
            Rew = torch.tensor(Rew, device=device)
            if use_teacher:
                w = sarsa_teacher(phi, w.detach(), S, A, Rew, args.gamma, args.alpha)
            else:
                H = build_prompt(phi, w.detach(), S, A, Rew, args.gamma)
                w = model(H, args.n).detach()
            s = int(S[args.n].item())
            step += 1
            if step % args.eval_every == 0:
                Q = phi.cpu().numpy() @ w.detach().cpu().numpy()
                rets[m, ei] = mc_return(mdp, greedy_probs_from_Q(Q), args.mc_trajs, args.mc_len, rng)
                ei += 1
    return rets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="eval")
    ap.add_argument("--nS", type=int, default=9)
    ap.add_argument("--nA", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--arch", choices=["linear", "softmax", "sarsa"], default="linear")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--fix-A", type=int, default=1)
    ap.add_argument("--n-mdps", type=int, default=100)
    ap.add_argument("--update-steps", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--mc-trajs", type=int, default=32)
    ap.add_argument("--mc-len", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.d is None:
        args.d = args.nS * args.nA

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 预生成一批 MDP + 特征，供 model/teacher/oracle/random 共用
    mdps = [sample_mdp(args.nS, args.nA, args.gamma, rng) for _ in range(args.n_mdps)]
    phis = [sample_features(args.nS, args.nA, args.d, rng) for _ in range(args.n_mdps)]

    if args.arch == "softmax":
        model = SoftmaxAttnICRL(d=args.d, temperature=args.temperature).to(device)
    elif args.arch == "sarsa":
        model = SoftmaxSARSA(d=args.d, temperature=args.temperature,
                             fix_A=bool(args.fix_A)).to(device)
    else:
        model = LinearAttnICRL(d=args.d).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ret_model = run_closed_loop(mdps, phis, False, model, args, rng, device)
    ret_teach = run_closed_loop(mdps, phis, True, model, args, rng, device)

    oracles, randoms = [], []
    for mdp in mdps:
        Q, _ = value_iteration(mdp)
        oracles.append(mc_return(mdp, greedy_probs_from_Q(Q), args.mc_trajs, args.mc_len, rng))
        uni = np.ones((mdp["nS"], mdp["nA"]), dtype=np.float32) / mdp["nA"]
        randoms.append(mc_return(mdp, uni, args.mc_trajs, args.mc_len, rng))
    oracle_mean = float(np.mean(oracles))
    random_mean = float(np.mean(randoms))

    xs = np.arange(1, args.update_steps // args.eval_every + 1) * args.eval_every
    out = {
        "update_steps": xs.tolist(),
        "ret_transformer": ret_model.mean(0).tolist(),
        "ret_teacher": ret_teach.mean(0).tolist(),
        "oracle": oracle_mean,
        "random": random_mean,
    }
    with open(os.path.join(args.outdir, "returns.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[model ] final %.4f   start %.4f" % (ret_model.mean(0)[-1], ret_model.mean(0)[0]))
    print("[teach ] final %.4f   start %.4f" % (ret_teach.mean(0)[-1], ret_teach.mean(0)[0]))
    print("[oracle] %.4f   [random] %.4f" % (oracle_mean, random_mean))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    plt.plot(xs, ret_model.mean(0), "o-", label="transformer")
    plt.plot(xs, ret_teach.mean(0), "s-", label="teacher")
    plt.axhline(oracle_mean, ls="--", color="k", label="oracle")
    plt.axhline(random_mean, ls=":", color="gray", label="random")
    plt.xlabel("in-context update step t")
    plt.ylabel("discounted return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "closed_loop.png"), dpi=150)
    print("[saved]", os.path.join(args.outdir, "closed_loop.png"))


if __name__ == "__main__":
    main()
