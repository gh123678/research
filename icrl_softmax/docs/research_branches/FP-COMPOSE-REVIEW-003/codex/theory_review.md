# 独立定理审查：T1–T5 与 K=12 的适用边界

作者：GPT；2026-09-15；审查基线 bcec1fe。此文件是对数学与实现依赖的事后独立重证，不以封存零违规或网络重放作为证明。数值重放 R 与扩维算术 D 因预审 O1/O2 暂停，不影响本项。

## 结论与限定

T1–T5、L12 单侧幅值界、Bellman 残差界与按状态条件性不退化：在下述理想采样/实数协议中 PASS。K=12 的风险为每条生产者-路线轨迹至多 0.15；不是整个 192 对路线矩阵的 95% 或 85% 联合保证。此结论不包括浮点软件的形式化舍入保证，也不把固定 PRNG 种子视作数学上随机独立的证明。

## 1. 直接核对的原始不等式

原始来源：Maurer–Pontil, Empirical Bernstein Bounds and Sample Variance Penalization, COLT 2009，Theorem 4，第 2 页：
https://www.cs.mcgill.ca/~colt2009/papers/012.pdf

原定理对 [0,1] 上 iid 样本给出单侧均值上偏差控制。其方差采用成对平方差形式，等于 ddof=1 样本方差。把区间缩放为长度 W 后，方差项为 sqrt(2*s²*log(2/delta)/n)，线性项为 7*W*log(2/delta)/(3*(n-1))。这一常数和方向与本组合实际使用一致；旧推导文档 §2 的双侧版本不应继续引用为正确命题，新的 FP-L12-one-sided-proof.md 才是有效替代。

## 2. T1：首访随机容量的完整论证

条件于环境、训练批、初始行为策略及该目标路线过去各步的认证数据 H，旧策略 pi 和当前 qhat 都固定。对指定状态动作 x，每条独立认证链 c 令 tau_c 为第一次取到 x 的时间（仅允许 0,...,L-1），I_c=1{tau_c<L}。链到 tau_c 的状态/动作已决定是否入选；残差使用的奖励和下一状态在该时间动作选择之后产生。

对每个 t 和可测集合 A，由 Markov 奖励/转移律：
Pr(tau_c=t, Y_c in A | H)=Pr(tau_c=t | H)*F_x(A)。
对 t 求和得到 Pr(I_c=1,Y_c in A | H)=Pr(I_c=1 | H)*F_x(A)。无需假设链内访问独立；只需要各条链独立、时间齐次的 x 条件转移/奖励律。没访问 x 的链中 Y_c 不必有定义，可以补一个独立的 F_x 标记来形式化“标记与入选独立”。

各链独立，故条件于任何入选指标向量 i，按链序排列的保留标记联合律为 F_x 的 N_x 次乘积。再在 sum(i)=n 的向量上混合，同样得到条件于 N_x=n 的 iid 标记。首访程序按全局索引排序在同一 x 内恰好保持链序。因此 MP 可以逐 n 使用并积分，无需对所有 n 额外 union。

不得进一步条件化于所有 x 的容量同时足够后宣称上述 iid：不同对的容量与残差可相关。正确做法定义单对坏事件 F_x={N_x>=2000 且 |rho_x|>eps_x}，逐 n 积分得 Pr(F_x|H)<=delta_dir，最后对 x union。若任一容量不足就弃权，错误发出事件是该 union 的子集。

实现追踪：evaluate_fp_xfam_001.py::vectorised_batch_generic 为固定行为策略生成独立链；fp_certfix_first_n.py::first_visit_batch 每条链每对只选首访；COMPOSE-002 evaluate.py 每步种子包含 step，生产者只接收 train 和历史 pi，raw 不传给生产者。认证批生成在生产者计算之前并不破坏数据依赖关系。

## 3. T2/T5：单侧幅值与包络

给定 H 与 N_x=n，rho_x 是固定分布参数。定义 E=R*+gamma*||Vhat||inf+||qhat||inf，残差 Y=r+gamma*Vhat(next)-qhat(x) 位于 [-E,E]；W=2E 为历史可测的有效区间长度。有限输出 qhat 任意，不需要其由 iid 训练数据估计；训练相关性不影响这一步。E 用当前 qhat 的范数，不使用当前认证批选择随机函数。

若 rho_x>0，MP 控制 rho_x-mean_x，故 |rho_x|<=mean_x+r_x<=|mean_x|+r_x；若 rho_x<0，对 -Y 使用同一定理，方差与 W 不变，得到同一输出上界；rho_x=0 时自动成立。方向由固定真实参数决定，而不是按样本选择方向，所以每对只消耗 delta_dir，不是 2*delta_dir。经验方差是 MP 本身允许的数据依赖统计量。

对 n 积分并对 d 个对 union，单步坏概率至多 d*delta_dir=delta_step。实际 L12 使用 n>=2000、ddof=1、delta_step/d、上述 W 和常数，未调用 L12M/L12S 传播。旧标量证明的双侧叙述已被撤回，不影响这份单侧证明。

## 4. 残差到 sup 误差

Qpi=Tpi Qpi；rho=Tpi qhat-qhat。Tpi 的线性部分为 gamma 倍随机矩阵，故
||qhat-Qpi||inf <= ||rho||inf + gamma*||qhat-Qpi||inf。
移项得 ||qhat-Qpi||inf <= max_x|rho_x|/(1-gamma) <= E_Q。需要 pi 各行是概率分布且 0<=gamma<1；理想相对 softmax 行归一化满足该条件。

## 5. T3：同一事件支持任意数据选择的行更新

在上述 sup 事件上，对任意行增量 u_s=pi_new(s)-pi_old(s)，Hölder 不等式给出
u_s dot Qpi(s) >= u_s dot qhat(s)-E_Q*||u_s||1 = LB_s。
此为对所有 u_s 同时成立的确定性命题，因此本步认证结果可以用于选 eta/更新行，不需要候选数的额外 union。

所有候选必须来自旧 pi；未通过行原样保持，故这些行的 u_s 为零。通过行 LB_s>0，得到整个优势向量 a=Tpi_new Vpi_old-Vpi_old>=0。性能差恒等式为
Vpi_new-Vpi_old=(I-gamma*Ppi_new)^(-1)*a。
逆矩阵是 sum_{t>=0} gamma^t*(Ppi_new)^t，逐元素非负，因此逐分量不退化。无需新策略下重新估计 qhat 后才选行，也无需对本步选行再做浓度估计。

检查 COMPOSE-002 evaluate.py::perstate_rows：候选始终取输入旧 policy；每行选第一个 LB>0 的 eta；未选行保留旧 policy。Qpi/vstar 只在审计使用，不进门限；按行条件证明适用。严格正改进只能直接保证所选行及能以新策略到达其优势的状态，不把逐分量严格改进当作无条件普遍命题。

## 6. 多步与停止

对一条路线的自然历史 H_(k-1)，当步 qhat/pi 固定而新批独立。定义坏事件为“该步实际检查且支持充分，但覆盖失败”；若已停止则该步坏事件为空。条件坏概率至多 delta_step，取期望并对最多 K 步 union 得 <=K*delta_step。本证明不条件化于“整条轨迹所有步都支持充分/仍在发出”，避免幸存者条件化。

K=4 用 0.05；K=12 保持 delta_step=0.0125，故为 0.15。两路线/两生产者共享同批不损害单条路线的此结论，但不产生矩阵联合保证。共享批跨路线执行次序不应进入单路线的数学条件历史；可在概念上定义各条完整路线，实际全局异常中止只删去部分发出，不增加坏更新。

## 7. T4 与软件边界

model.py 的 EndToEndMaskedSoftmaxExpectedSARSA 和 EndToEndFiniteSoftmaxExpectedSARSA 构造函数没有训练参数，forward 只读 q、训练转移、pi 与固定常数。COMPOSE-002 以零 q 开始实际调用 160 次，没有优化器、反向传播、oracle 修正。网络负责 qhat；认证与更新仍在外部完成。这是被测网络实现的身份检查，不能仅凭其类名证明一般 softmax 模型的策略改进能力。

理想概率证明假设新鲜独立随机数；NumPy 的预先固定 seed schedule 保证可复现与无显式复用，却不是“已固定种子下仍有 85% 概率”的数学主张。浮点 CDF、行和归约与 float32/64 都需要数值审计，不能由本证明升级为严格机器证书。上述限制是原任务已声明的实数协议边界，不是本次发现 L12 无效。

## 分层判定

T1 首访与历史条件 PASS；T2 单侧幅值/包络 PASS；T3 条件性不退化 PASS；T4 实现依赖隔离 PASS（尚未用本次实际前向重放核实数值）；T5 数据依赖函数误用检查 PASS。R 尚未执行，不归入本判定。

PASS
