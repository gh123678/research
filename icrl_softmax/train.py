"""IC-SARSA 训练（03 论文 Algorithm 1）+ 参数结构涌现检查。

用法示例（sanity check）：
    python train.py --K 300 --T 30 --outdir out_sanity
完整规模（论文配置，很慢）：
    python train.py --K 10000 --T 1000 --outdir out_full
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mdps import sample_mdp, sample_features, eps_greedy_probs, rollout
from model import LinearAttnICRL, SoftmaxAttnICRL


def build_prompt(phi, w, S, A, Rew, gamma):
    """03 论文式 4：H in R^(D x (n+1))，D = 3d + 2。phi:(nS,nA,d)。"""
    d = phi.shape[-1]
    D = 3 * d + 2
    n = len(Rew) - 1
    H = torch.zeros(D, n + 1, dtype=phi.dtype, device=phi.device)
    H[:d, :n] = phi[S[:n], A[:n]].t()                       # phi_i
    H[d:2 * d, :n] = gamma * phi[S[1:n + 1], A[1:n + 1]].t()  # gamma * phi^+_i
    H[2 * d, :n] = Rew[1:]                                  # r_{i+1}
    wt = torch.zeros(d + 1, dtype=phi.dtype, device=phi.device)
    wt[0] = 1.0
    wt[1:] = w
    H[2 * d + 1:, n] = wt                                   # w_tilde
    return H


def sarsa_teacher(phi, w, S, A, Rew, gamma, alpha):
    """03 论文式 2 的 batch 半梯度 SARSA 更新（teacher 目标）。"""
    n = len(Rew) - 1
    phi_all = phi[S, A]                       # (n+1, d)
    td = Rew[1:] + gamma * (phi_all[1:] @ w) - (phi_all[:-1] @ w)  # (n,)
    grad = (td.unsqueeze(1) * phi_all[:-1]).sum(0)                 # (d,)
    return w + (alpha / n) * grad


def get_blocks(model):
    """取有效块 (P12, V21)。full 模式下从完整矩阵切出对应子块。"""
    if hasattr(model, "P12"):
        return model.P12.detach().cpu().numpy(), model.V21.detach().cpu().numpy()
    P, V = model._assemble()
    P = P.detach().cpu().numpy()
    V = V.detach().cpu().numpy()
    d = model.d
    return P[: 2 * d + 1, 2 * d + 1:], V[2 * d + 2:, : 2 * d + 1]


def check_structure(model, alpha, d, outdir):
    """检查学到的 (P12, V21) 是否涌现出理论结构（V21 左上块对角化）。"""
    P12, V21 = get_blocks(model)
    top = V21[:, :d]                                  # 期望 ~ c*alpha*I_d
    diag = float(np.abs(np.diag(top)).mean())
    off = top - np.diag(np.diag(top))
    offdiag = float(np.abs(off).mean())
    ratio = diag / max(offdiag, 1e-9)
    scale = float(diag / alpha) if alpha > 0 else float("nan")
    print("[structure] V21[:,:d]  |diag|=%.4f |offdiag|=%.4f  ratio=%.2f (c≈%.3f)"
          % (diag, offdiag, ratio, scale))
    # 理论对齐余弦：V21[:,:d] vs alpha*I_d（去缩放）
    norm_v = top / (np.linalg.norm(top) + 1e-9)
    norm_t = np.eye(d) / np.linalg.norm(np.eye(d))
    cos = float(np.sum(norm_v * norm_t))
    print("[structure] V21[:,:d] vs alpha*I_d  cosine=%.4f" % cos)
    return {"V_diag_ratio": ratio, "V_cos_alphaI": cos}


def save_heatmaps(model, d, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P12, V21 = get_blocks(model)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axs[0].imshow(P12, cmap="RdBu_r", aspect="auto")
    axs[0].set_title("learned P12  (2d+1, d+1)")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(V21, cmap="RdBu_r", aspect="auto")
    axs[1].set_title("learned V21-bar  (d, 2d+1)")
    fig.colorbar(im1, ax=axs[1])
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "params.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nS", type=int, default=9)
    ap.add_argument("--nA", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--d", type=int, default=None, help="特征维数，默认 nS*nA")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.2, help="SARSA step size")
    ap.add_argument("--n", type=int, default=20, help="轨迹窗口长度")
    ap.add_argument("--arch", choices=["linear", "softmax"], default="linear")
    ap.add_argument("--freeze-inert", type=int, default=1, help="softmax 是否冻结惰性子块")
    ap.add_argument("--temperature", type=float, default=1.0, help="softmax 得分温度，训练/评估须一致")
    ap.add_argument("--flip-check-interval", type=int, default=0, help="每 N 个 MDP 检查 V21 符号，0 关闭")
    ap.add_argument("--flip-max", type=int, default=3, help="最多取反次数")
    ap.add_argument("--K", type=int, default=300, help="MDP 数量")
    ap.add_argument("--T", type=int, default=30, help="每个 MDP 的帧数")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-decay", type=float, default=0.99)
    ap.add_argument("--lr-decay-interval", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="out")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    if args.d is None:
        args.d = args.nS * args.nA

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    if args.arch == "softmax":
        model = SoftmaxAttnICRL(d=args.d, freeze_inert=bool(args.freeze_inert),
                                temperature=args.temperature).to(device)
    else:
        model = LinearAttnICRL(d=args.d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_hist = []
    for k in range(args.K):
        mdp = sample_mdp(args.nS, args.nA, args.gamma, rng)
        phi = torch.tensor(sample_features(args.nS, args.nA, args.d, rng), device=device)
        w = torch.tensor(rng.uniform(-1, 1, size=args.d).astype(np.float32), device=device)
        s = int(rng.choice(args.nS, p=mdp["p0"]))
        for t in range(args.T):
            probs = eps_greedy_probs(phi.cpu().numpy(), w.detach().cpu().numpy(), args.eps)
            S, A, Rew = rollout(mdp, probs, s, args.n, rng)
            S = torch.tensor(S, device=device)
            A = torch.tensor(A, device=device)
            Rew = torch.tensor(Rew, device=device)
            H = build_prompt(phi, w, S, A, Rew, args.gamma)
            w_pred = model(H, args.n)
            w_target = sarsa_teacher(phi, w.detach(), S, A, Rew, args.gamma, args.alpha)
            loss = 0.5 * F.mse_loss(w_pred, w_target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_hist.append(loss.item())
            w = w_target.detach()
            s = int(S[args.n].item())
        if (k + 1) % args.lr_decay_interval == 0:
            for g in opt.param_groups:
                g["lr"] *= args.lr_decay
        if (args.flip_check_interval > 0 and hasattr(model, "V21")
                and (k + 1) % args.flip_check_interval == 0):
            top = model.V21.detach()[:, : args.d]
            sign = float(torch.diag(top).mean())
            if sign < 0:
                with torch.no_grad():
                    model.V21.data.mul_(-1.0)
                print("[flip] MDP %d  V21 diag sign %.3f -> flipped" % (k + 1, sign))
        if (k + 1) % args.log_every == 0 or k + 1 == args.K:
            cur = float(np.mean(loss_hist[-args.T:]))
            print("[MDP %d/%d] frame-loss %.6f  lr %.2e" %
                  (k + 1, args.K, cur, opt.param_groups[0]["lr"]))

    np.save(os.path.join(args.outdir, "loss.npy"), np.array(loss_hist))
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pt"))
    stats = check_structure(model, args.alpha, args.d, args.outdir)
    save_heatmaps(model, args.d, args.outdir)
    with open(os.path.join(args.outdir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("[done] saved to", args.outdir)


if __name__ == "__main__":
    main()
