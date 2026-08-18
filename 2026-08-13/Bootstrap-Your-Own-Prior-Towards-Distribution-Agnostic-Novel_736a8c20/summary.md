---
title: "Bootstrap-Your-Own-Prior-Towards-Distribution-Agnostic-Novel"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Bootstrap_Your_Own_Prior_Towards_Distribution-Agnostic_Novel_Class_Discovery_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:40:49"
field: "开放世界视觉识别"
keywords: ["Novel Class Discovery", "Distribution-Agnostic", "Imbalanced Learning", "Optimal Transport", "Self-supervised Clustering", "Pseudo-label"]
innovations: ["提出distribution-agnostic NCD任务，放松均匀分布假设", "设计BYOP迭代先验估计与动态温度技术解决长尾NCD", "将最优传输聚类与在线先验估计结合实现自适应伪标签生成"]
benchmarks: ["CIFAR10", "CIFAR100-20", "CIFAR100-50", "Tiny-ImageNet"]
---

# 论文速读：Bootstrap-Your-Own-Prior-Towards-Distribution-Agnostic-Novel

## 一句话总结
本文针对 Novel Class Discovery（NCD）中新类别分布不平衡的现实问题，提出了 **distribution-agnostic NCD** 新任务与 **BYOP（Bootstrapping Your Own Prior）** 方法，通过迭代自估计类别先验 + 动态温度技术，显著提升了在任意未知类别分布下的新类别发现性能。

## 研究问题与动机
1. **现实分布不平衡被忽视**：现有 NCD 方法普遍假设未标注的新样本服从均匀类别分布（uniform prior），但在真实世界大规模数据中类别往往高度不平衡。
2. **均匀先验在长尾场景下有害**：图 1 说明，若数据实际不平衡却强加均匀先验，聚类会将多数类样本错误分散到多个簇，导致性能下降甚至适得其反。
3. **"鸡生蛋"困境**：类别先验是聚类的关键约束，但先验本身又是未知的，需依赖模型预测来估计，形成循环依赖。
4. **已有方法难以迁移**：基于成对相似度（RS、NCL）或显式均匀正则化（UNO、ComEx）的方法在分布-不可知设定下表现显著退化。

## 核心贡献（创新点）
1. **提出 distribution-agnostic NCD 新任务**：放松均匀分布假设，允许新样本来自任意未知类别分布，使现有方法在真实场景下失效。
2. **设计 BYOP 迭代训练范式**：通过模型自身预测估计类别先验 → 生成更准确伪标签 → 促进下一轮预测，打破"鸡生蛋"困境。
3. **发明动态温度（Dynamic Temperature）技术**：按样本置信度自适应调整 softmax 温度，对低置信样本施加更大温度以放大 CE loss，从而推动更确定的预测。
4. **提出在线轻量先验估计**：利用 FIFO 队列记录最近 batch 的 logits，配合移动平均更新先验，无需额外前向传播，计算开销极小。
5. **与长尾学习技术可组合**：利用估计出的先验，可与 Logit Adjustment 等后处理策略结合，进一步改善少数类性能。

## 方法详解

### 整体架构
- 共享图像编码器 $\phi(\cdot)$（ResNet-18）提取特征 $z = \phi(x)$，特征 $\ell_2$ 归一化。
- 基础类头 $h(\cdot)$：$C^b$ 类线性分类器；新类别头 $g(\cdot)$：MLP 降维后接 $C^n$ 类线性分类器。
- 统一训练目标：将 base 和 novel 的 logits 拼接为 $q = [q^b, q^n] \in \mathbb{R}^{C^b+C^n}$，使用统一 cross-entropy loss。

### 3.1 带类别先验的聚类（Clustering with Class Prior）
- 将 novel 头 $g(\cdot)$ 的权重矩阵 $W$ 视为簇中心，用**最优传输**（Optimal Transport）将样本分配至簇。
- 优化目标（公式 1）：
  $$\max_{Y \in \mathcal{T}} \text{tr}(Y^\top W^\top X) + \epsilon H(Y)$$
- 约束集合（公式 2）：$Y \mathbf{1}_B = p$，即每个簇的分配比例由先验向量 $p$ 控制（而非固定均匀值）。
- 用 **Sinkhorn-Knopp 算法**求解，得到软概率伪标签 $Y$（非 hard one-hot，适合小 batch 训练）。
- 初始化时 $p = \frac{1}{C^n}\mathbf{1}$，后续迭代用估计先验更新。

### 3.2 类别分布预测 + 动态温度
- **标准 CE loss**（公式 3）：$\mathcal{L} = -\sum_c y_c \log(\hat{y}_c),\; \hat{y} = \sigma(q/\tau)$，其中 $\tau=0.1$ 为默认温度。
- **动态温度**（公式 4）：$\tau' = \tau / \rho,\; \rho = \max(\sigma(q/\tau))$。
  - 高置信样本：$\rho \approx 1$，$\tau' \approx \tau$，几乎不影响原有学习。
  - 低置信样本：$\rho$ 较小，$\tau'$ 增大 → softmax 输出更平坦 → CE loss 更大 → 迫使模型产生更确定预测。
- 图 3 证明：动态温度显著减少了少数类的预测混淆。

### 3.3 在线类别先验估计
- 维护大小为 $K=6000$ 的 FIFO 队列 $\mathcal{K}$，记录每 batch 中 novel 样本的 logits。
- 先验估计（公式 6）：对队列中每个样本取 $\arg\max q_c^n$ 作为其类别分配，统计各类别占比得 $r$。
- 移动平均更新（公式 7）：$p \leftarrow \mu p + (1-\mu)r$，其中 $\mu=0.99$。
- 延迟 5 个 epoch 后再启动先验估计，保证初始阶段稳定性。
- 整个流程见 Algorithm 1。

## 实验与结果

### 数据集与设置
- **数据集**：CIFAR10、CIFAR100-20、CIFAR100-50、Tiny-ImageNet（表 1）。
- **不平衡比率**：10 和 100，分 many/medium/few 三档。
- **评估协议**：Traditional（训练集）、Task-aware、Task-agnostic（测试集），使用匈牙利算法求最优聚类匹配（Cluster Acc）。
- **实现细节**：ResNet-18 编码器，预训练 200 epoch + 联合训练 200 epoch；$\tau=0.1$，$|K|=6000$，$\mu=0.99$，$\epsilon=0.05$，Sinkhorn 迭代 3 次。

### 主要结果
- **消融实验（表 2，CIFAR100-50，imbalance=100）**：Estimated p 优于 Uniform p（All: 29.4 vs 25.7）；Oracle p 与 Estimated p 接近，验证估计有效性；动态温度在各条件下均带来提升。
- **CIFAR10 imbalance=100（表 3）**：UNO → UNO+BYOP，传统协议 Novel 聚类准确率从 43.9% → **59.3%**（+15.4%）；ComEx → ComEx+BYOP，从 44.6% → **57.0%**（+12.4%）。
- **CIFAR10 imbalance=10（表 3）**：UNO+BYOP 达到 63.6%（Nov. Trad.），较 UNO 的 59.6% 提升 4.0%。
- **CIFAR100-20/50（表 4-5）**：BYOP 在各不平衡比率和协议下均显著超越基线，尤其在 Traditional 协议上提升最大。
- **Tiny-ImageNet（表 6）**：因类别数多（100），提升相对较小，但仍为正增长，说明任务本身更具挑战性。
- **最强结果**：CIFAR10 imbalance=100 下 UNO+BYOP 传统协议 Novel 聚类准确率 **59.3%**，较 SOTA UNO 提升约 15 个百分点。

## 相关工作脉络
1. **Han et al. (RS/RS+, ICLR 2020)**：基于成对相似度排名的 NCD 方法，不显式施加均匀约束，在长尾下仍有一定鲁棒性，但不如 BYOP。
2. **Zhong et al. (NCL, CVPR 2021)**：邻域对比学习用于 NCD，同样依赖自监督预训练和强数据增强，性能接近但实现更复杂。
3. **Fini et al. (UNO, ICCV 2021)**：统一目标 NCD，使用 Sinkhorn 聚类并施加均匀正则化，是本工作最直接的对比基线；BYOP 在其框架上叠加先验估计和动态温度即显著提升。
4. **Yang et al. (ComEx, CVPR 2022)**：通用化 NCD，引入 compositional expert，同样假设均匀分布；BYOP 可无缝集成于其训练范式。
5. **Asano et al. (Self-labeling, ICLR 2020)** 与 **Caron et al. (DINO, NeurIPS 2020)**：最优传输聚类的先驱工作，BYOP 沿用了其 Sinkhorn-based 聚类思路但将先验从固定均匀改为自适应估计。
6. **Menon et al. (Logit Adjustment, ICLR 2021)**：长尾学习的经典后处理技术，本文证明其与 BYOP 估计的先验可结合（表 7），为后续研究提供组合思路。

## 局限性与未来方向
1. **少数类性能仍有瓶颈**：即使使用 BYOP，极端不平衡下的 few-shot 类别聚类准确率仍偏低（如 CIFAR100-50 imbalance=100 下 Few 仅 12.4%~14.1%），视觉上（图 4）也显示少数类簇划分不够清晰。
2. **类别数较多时提升有限**：Tiny-ImageNet（100 个 novel 类）上 BYOP 的增益幅度明显缩小，说明在高维复杂场景下分布-不可知 NCD 仍是开放挑战。
3. **仅假设 $C^n$ 已知**：当前设定依赖预先知道新类别数量，若 $C^n$ 未知则需额外机制（论文附录简要提及但未深入）。
4. **动态温度的理论分析不足**：方法的经验效果显著，但缺乏对温度自适应策略收敛性或泛化边界的严格理论支撑。
5. **未探索跨域/多模态场景**：扩展至多域 NCD 或多模态 NCD（如 CLIP 风格）的潜力未被挖掘。

## 研究启发与可借鉴点
1. **先验估计的迭代自举思想可迁移**：在任意聚类/划分任务中，若类别比例未知，可用"预测→估计→再聚类"的循环范式，适用于半监督学习、开放集识别等场景。
2. **动态温度作为一种通用置信度校准技巧**：对软标签或不确定样本自动增大温度以放大学习信号，可嵌入任意基于 softmax 的分类/聚类训练流程，尤其适合伪标签训练。
3. **FIFO 队列 + 移动平均的在线先验更新**：无需额外前向传播即可实时估计分布，计算开销极小，是工程中轻量且有效的实现方式。
4. **BYOP 与长尾学习技术的正交性**：本文展示的先验估计与 Logit Adjustment 的组合，提示可将 NCD 与长尾分类、重加权、解耦训练等技术深度结合，开辟新方向。
5. **最优传输聚类的先验可控性**：Sinkhorn 约束中的 $p$ 向量直接控制簇大小，这一灵活接口可在多个 unsupervised/semi-supervised 设定中复用，值得系统性探索。

## 关键术语表
- **Novel Class Discovery (NCD)**：利用已标注基础类的知识，在无标注的新数据上自动发现并聚类未知新类别的迁移学习任务。
- **Distribution-agnostic NCD**：本文提出的新任务设定，新样本可来自任意未知类别分布，不再假设均匀分布先验。
- **BYOP (Bootstrapping Your Own Prior)**：本文方法，通过迭代自估计类别先验来指导聚类，逐步优化伪标签质量。
- **Optimal Transport Clustering**：基于最优传输理论的聚类方法，通过 Sinkhorn-Knopp 算法求解带约束的样本-簇分配。
- **Sinkhorn-Knopp 算法**：求解 entropic-regularized 最优传输问题的迭代投影算法，本文用于带类别先验约束的软聚类。
- **Dynamic Temperature**：按样本预测置信度自适应缩放 softmax 温度的技术，低置信样本获得更大温度以增强学习信号。
- **Task-aware / Task-agnostic Protocol**：前者测试时提供样本属于 base/novel 的任务信息；后者不提供，更具挑战性。
- **Logit Adjustment**：针对类别不平衡的后处理技术，通过对 logit 减去 $\tau \log(p)$ 来补偿先验偏差。

## 可复现要素
- **数据集**：CIFAR10、CIFAR100、Tiny-ImageNet 均为公开数据集。
- **代码**：论文未明确声明代码开源（CVPR 2023 当时未强制开源），但作者在文中多次引用已开源项目（UNO、ComEx 等），复现可在此基础上添加 BYOP 模块。
- **权重**：使用与 UNO/ComEx 相同的预训练权重（论文未提供单独权重下载链接）。
- **关键超参**：温度 $\tau=0.1$，队列大小 $|K|=6000$，移动平均因子 $\mu=0.99$，Sinkhorn 正则化 $\epsilon=0.05$，迭代次数 3，延迟 5 epoch 启动先验估计，预训练 200 epoch + 联合训练 200 epoch，编码器 ResNet-18。
