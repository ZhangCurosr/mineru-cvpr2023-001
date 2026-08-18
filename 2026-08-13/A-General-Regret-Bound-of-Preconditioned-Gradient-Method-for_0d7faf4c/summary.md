---
title: "A-General-Regret-Bound-of-Preconditioned-Gradient-Method-for"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Yong_A_General_Regret_Bound_of_Preconditioned_Gradient_Method_for_DNN_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:48:45"
field: "深度学习优化算法"
keywords: ["DNN优化器", "预条件梯度", "遗憾界", "Kronecker因子化", "自适应学习率", "AdaBK", "计算机视觉"]
innovations: ["建立约束满矩阵预条件梯度的通用遗憾界理论，将遗憾界最小化转化为引导函数最小化", "在分块对角和Kronecker因子化约束下严格推导更新公式，提出AdaBK优化器", "设计Schur-Newton矩阵逆根、低频统计更新、自适应dampening和梯度范数恢复四组实用化技术"]
benchmarks: ["CIFAR-100", "CIFAR-10", "ImageNet-1k", "COCO"]
---

# 论文速读：A-General-Regret-Bound-of-Preconditioned-Gradient-Method-for-DNN-Training

## 一句话总结
本文建立了约束满矩阵预条件梯度方法的通用遗憾界（regret bound）理论，证明通过在锥约束下最小化引导函数可同时最小化遗憾界，并据此推导出新的 DNN 优化器 AdaBK（嵌入 SGDM/AdamW 后称 SGDM_BK / AdamW_BK），在图像分类、目标检测和分割任务上显著优于现有主流优化器，仅增加 10%~25% 训练时间。

## 研究问题与动机
1. **全矩阵预条件梯度的遗憾界更优但不可行**：AdaGrad 等自适应学习率方法仅利用梯度二阶信息的对角元，而全矩阵预条件梯度 $H_T^{-1}g_T$（其中 $H_T = (\sum_t g_t g_t^\top)^{1/2}$）具有更低的遗憾界，但因参数空间维度极高而无法用于 DNN 训练。
2. **现有约束满矩阵方法缺乏理论指导**：GGT、Shampoo、KFAC 等工作通过人为设计的结构约束（如对角、分块对角、Kronecker 分解）近似 $H_T$，但这些方法是启发式的，其代价函数对遗憾界的影响未知。
3. **需要统一的理论框架来指导预条件器设计**：尚无线上的遗憾界理论能够指导不同约束下满矩阵预条件梯度方法的设计，以及对应的更新公式推导。

## 核心贡献（创新点）
1. **提出约束满矩阵预条件梯度的通用遗憾界定理（Theorem 1）**：将锥约束 $\Psi$ 下的遗憾界与引导函数 $F_T(S) = \sum_t \|g_t\|_S^*$ 关联，证明最小化该引导函数可同时最小化遗憾界。与已有工作的本质区别：首次给出约束满矩阵方法的遗憾界理论，此前的相关方法（如 KFAC、Shampoo）均为启发式设计，缺乏理论保证。
2. **在分块对角与 Kronecker 因子化约束下推导具体更新公式（AdaBK）**：分别导出 layer-wise block-diagonal 和 Kronecker-factorized 约束下的引导函数及优化解，得出更新规则 $W_{t+1} = W_t - \eta L_t^{-1/2} G_t R_t^{-1/2}$。与已有工作的本质区别：基于理论推导而非经验设计，且推导出的更新形式与 Shampoo/KFAC 形式不同（后者基于 Fisher 矩阵近似）。
3. **提出四组工程加速技术使 AdaBK 可实用**：包括 Schur-Newton 矩阵逆根计算（避免低效 SVD）、统计量指数滑动平均更新（低频更新 $L_t/R_t$ 及其逆根）、自适应 dampening（$\epsilon \lambda_{max}$ 确保正定性）、梯度范数恢复（保持与原优化器相同的梯度幅度）。与已有工作的本质区别：系统性地解决满矩阵方法在高维 DNN 训练中的数值稳定性与计算效率问题。
4. **验证 SGDM_BK 和 AdamW_BK 在多类视觉任务上的有效性**：在 CIFAR、ImageNet、COCO 上的分类、检测、分割任务中均取得显著性能提升，且可直接复用原优化器的超参。

## 方法详解
1. **通用遗憾界（Theorem 1）**：给定锥约束集 $\Psi$，定义引导函数 $F_T(S) = \sum_{t=1}^T \|g_t\|_S^*$，令 $S_T = \arg\min_{S \in \Psi, S \succeq 0, \text{Tr}(S)\leq 1} F_T(S)$，$C_T = \sqrt{F_T(S_T)}$，则遗憾界满足：
   $$R(T) \leq \left(\frac{D^2}{2\eta} + \eta\right) \sqrt{\min_{S \in \Psi, S \succeq 0, \text{Tr}(S)\leq 1} F_T(S)}$$
   即最小化引导函数的上界等价于最小化遗憾界。当 $\Psi_1 \subseteq \Psi_2$ 时，约束越小（越宽松），遗憾界越紧，解释全矩阵最优。

2. **Kronecker 因子化约束下的推导**：对 FC 层，设 $S = S_1 \otimes S_2$，$S_1 \in \mathbb{R}^{C_{out}\times C_{out}}$，$S_2 \in \mathbb{R}^{C_{in}\times C_{in}}$，引导函数被放缩为：
   $$F_T(S) \leq \frac{1}{n}\text{Tr}(S_1^{-1} L_T) \cdot \text{Tr}(S_2^{-1} R_T)$$
   其中 $L_T = \sum_{t,i} \delta_{ti}\delta_{ti}^\top$（输出特征梯度外积之和），$R_T = \sum_{t,i} x_{ti}x_{ti}^\top$（输入特征外积之和）。经 Lemma 4 求解得：
   $$H_{1,T} = L_T^{1/2}, \quad H_{2,T} = R_T^{1/2}, \quad H_T = H_{1,T} \otimes H_{2,T}$$
   更新公式为 $W_{t+1} = W_t - \eta L_t^{-1/2} G_t R_t^{-1/2}$。

3. **Conv 层扩展**：通过 im2col 将卷积操作等价为矩阵乘法，权重经 mode-1 unfold 后可视为 FC 层处理，计算流程相同。

4. **Schur-Newton 矩阵逆根计算**：替代低效的 SVD，通过迭代（$K=10$ 次）计算 $A^{-1/2} \approx Z_K / \sqrt{\text{Tr}(A)}$，其中：
   $$T_k = \tfrac{1}{2}(3I - Z_{k-1}Y_{k-1}), \quad Y_k = Y_{k-1}T_k, \quad Z_k = T_k Z_{k-1}$$

5. **统计量低频更新**：引入超参 $T_s$（统计量更新频率）和 $T_{ir}$（逆根更新频率），采用指数滑动平均 $L_t = \alpha L_{t-1} + (1-\alpha)\Delta_t\Delta_t^\top$，$T_s=200$、$T_{ir}=2000$，显著降低计算开销。

6. **自适应 dampening**：添加 $\epsilon \lambda_{max} I$（$\lambda_{max}$ 由 power iteration 求得），条件数被有界控制在 $\frac{1+\epsilon}{\epsilon}$，增强数值稳定性。

7. **梯度范数恢复**：对预条件梯度 $\widehat{G}_t = L_t^{-1/2} G_t R_t^{-1/2}$ 乘以缩放因子使其 $L_2$ 范数与原始梯度一致，从而可直接复用原优化器（SGDM/AdamW）的超参。

## 实验与结果
- **数据集**：CIFAR-100、CIFAR-10、ImageNet-1k（分类）；COCO（检测 Faster-RCNN、分割 Mask-RCNN）
- **评估基线**：SGDM、AdamW、AdaGrad、RAdam、Adabelief、Shampoo、KFAC、WSGDM
- **主要结果（分类）**：
  - CIFAR-100 / ResNet50：SGDM_BK = 81.26%（↑3.48% over SGDM），AdamW_BK = 80.15%（↑2.05% over AdamW）
  - CIFAR-100 / VGG19：SGDM_BK = 75.10%（↑4.16%），AdamW_BK = 74.27%（↑4.01%）
  - ImageNet-1k / ResNet50：SGDM_BK = 77.62%（↑1.31%），AdamW_BK = 77.22%（↑1.10%）
  - ImageNet-1k / Swin-T：AdamW_BK = 81.79%（↑0.61%）
- **主要结果（检测/分割 COCO）**：
  - Faster-RCNN R50：SGDM_BK AP↑2.2%，AdamW_BK AP↑1.6%
  - Mask-RCNN R50：SGDM_BK AP^b↑2.2%、AP^m↑2.2%；AdamW_BK AP^b↑2.2%、AP^m↑1.3%
  - Mask-RCNN Swin-T 1X：AdamW_BK AP^b↑0.9%、AP^m↑0.9%
- **最强结果**：CIFAR-100 VGG19 上 SGDM_BK 取得 75.10% 准确率，相对 SGDM 提升 **4.16%**；COCO Faster-RCNN R50 上 SGDM_BK 取得 **39.6 AP**，相对 SGDM 提升 **2.2%**。
- **计算开销**：额外内存约 650-670 MiB（相对基线增加约 1-2%），额外训练时间 **10%~25%**。

## 相关工作脉络
1. **AdaGrad [5]**：仅使用梯度的对角二阶统计量进行自适应学习率更新；本文将其视为锥约束 $\Psi$ 为对角矩阵时的特例，遗憾界更宽松。
2. **Shampoo [9]**：使用分块对角（layer-wise full-matrix）预条件器；本文与其形式相似但理论出发点不同——Shampoo 为启发式构造，本文通过引导函数最小化严格推导。
3. **KFAC [7]**：使用 Kronecker 因子化的 Fisher 信息矩阵近似；本文同样采用 Kronecker 因子化约束，但基于梯度外积而非 Fisher 矩阵，且附带遗憾界理论保证。
4. **GGT [1]**：存储近期梯度低秩近似以加速满矩阵预条件计算；本文不依赖低秩近似，而是通过结构约束直接降维。
5. **WSGDM [33]**：嵌入式特征白化方法，通过梯度归一化改善训练；本文的梯度范数恢复技术在动机上与 WSGDM 相似，但理论基础（遗憾界最小化）不同。
6. **Natural Gradient [6,7]**：使用 Fisher 矩阵的近似作为预条件器；本文方法不使用 Fisher 矩阵，而是直接使用梯度外积的 Kronecker 分解，计算更轻量。

## 局限性与未来方向
1. **计算开销仍高于自适应学习率方法**：虽然引入低频更新策略将额外开销控制在 10%~25%，但在大规模模型（如 ViT-Large、LLM）上，Schur-Newton 矩阵逆根计算可能仍面临挑战。
2. **仅针对卷积和 FC 层验证**：实验主要集中在 Vision 任务，未涉及 NLP、强化学习等其他领域，跨领域泛化能力有待验证。
3. **引导致函数的上界放缩可能损失精度**：Lemma 3 中对 $F_T(S)$ 的放缩（从 $\frac{1}{n}\sum g_{ti}g_{ti}^\top$ 到 $L_T, R_T$ 的外积分离）是一种保守近似，未来可探索更紧的放缩或直接用原始引导函数优化。
4. **未讨论收敛到局部最优的理论分析**：遗憾界是针对 online convex optimization 框架的上界，对于非凸 DNN 训练，其实际意义的理论分析有待进一步研究。

## 研究启发与可借鉴点
1. **引导函数最小化作为预条件器设计的统一框架**：将遗憾界与引导函数关联的思路可迁移至其他结构约束（如低秩、稀疏结构）下的预条件器设计，形成"理论驱动+结构约束"的方法论。
2. **梯度范数恢复技巧可直接复用**：该技巧使新优化器可直接继承现有优化器（SGDM/AdamW）的成熟超参，大幅降低实际部署难度，值得推广到其他新优化器的设计中。
3. **Schur-Newton 替代 SVD 的矩阵根计算方法**：该方法高效且可 GPU 并行，可作为深度学习框架中矩阵运算的新工具，适用于其他需要矩阵逆根的场景（如 Whitening、协方差正则化）。
4. **锥约束与遗憾界单调性的关系**：$\Psi_1 \subseteq \Psi_2 \Rightarrow$ 约束越宽松遗憾界越小的结论，为不同预条件器结构的比较提供了理论依据，可用于分析各类优化器的理论性能上界。

## 关键术语表
**遗憾界（Regret Bound）**：在线优化中衡量算法累积损失与最优固定策略累积损失之差的上界，值越小表示优化效果越好。
**预条件梯度（Preconditioned Gradient）**：用正定矩阵 $H_t^{-1}$ 对梯度进行左乘变换后的更新方向，可自适应调整各参数的学习尺度与相关性。
**引导函数（Guide Function）**：$F_T(S) = \sum_t \|g_t\|_S^*$，其最小化可同时最小化预条件梯度方法的遗憾界，是本文理论推导的核心桥梁。
**Kronecker 因子化约束**：将大矩阵预条件器分解为两个较小矩阵的 Kronecker 积 $S = S_1 \otimes S_2$，大幅降低参数量，KFAC 和 Shampoo 均采用此思路。
**Schur-Newton 算法**：一种用于计算矩阵平方根/逆根的迭代算法，相比 SVD 更适合 GPU 实现，收敛速度快且数值稳定。
**锥约束（Cone Constraint）**：集合 $\Psi$ 满足对任意 $x \in \Psi$ 和 $\theta > 0$ 有 $\theta x \in \Psi$，是本文遗憾界定理成立的关键前提条件。
**梯度范数恢复（Gradient Norm Recovery）**：将预条件梯度缩放至与原梯度相同的 $L_2$ 范数，使超参可直接迁移，避免重新调参。

## 可复现要素
- **数据集**：CIFAR-100、CIFAR-10、ImageNet-1k、COCO（均为公开数据集）
- **代码开源**：是，仓库地址 https://github.com/Yonghongwei/AdaBK
- **框架**：PyTorch 1.11
- **关键超参**：$\alpha = 0.9$（滑动平均系数），$T_s = 200$（统计量更新频率），$T_{ir} = 2000$（逆根更新频率），$\epsilon = 0.00001$（dampening），Schur-Newton 迭代次数 $K=10$，power iteration 次数 $K=10$
- **实验硬件**：NVIDIA GeForce RTX 2080 Ti / 3090 Ti GPU
