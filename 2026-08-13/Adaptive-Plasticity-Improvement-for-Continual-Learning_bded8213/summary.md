---
title: "Adaptive-Plasticity-Improvement-for-Continual-Learning"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Liang_Adaptive_Plasticity_Improvement_for_Continual_Learning_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:49:42"
field: "持续学习"
keywords: ["continual learning", "catastrophic forgetting", "adaptive plasticity", "gradient projection", "dual gradient projection memory"]
innovations: ["提出DualGPM通过维护梯度子空间及其正交补中较小者实现内存先增后减", "首次定义AGRR量化评估模型可塑性并与准确率强相关", "基于GRR自适应扩展输入维度以按需提升可塑性"]
benchmarks: ["Split CIFAR100", "CIFAR100-sup", "Split Mini-Imagenet", "5-Datasets"]
---

# 论文速读：Adaptive-Plasticity-Improvement-for-Continual-Learning

## 一句话总结
论文提出自适应可塑性改进（API）方法，通过引入DualGPM克服灾难性遗忘，并定义梯度保留率（GRR/AGRR）量化评估模型可塑性，在可塑性不足时自适应扩展输入维度以提升新任务学习能力。

## 研究问题与动机
- **可塑性被忽视**：现有持续学习方法（正则化、记忆、扩展类）主要追求稳定性（克服灾难性遗忘），但缺乏对模型可塑性的评估与改进机制。
- **约束累积导致可塑性下降**：随着任务数增加，正则化/投影类方法的约束越强，模型对新任务的参数调整能力逐渐衰减。
- **扩展方法未量化评估**：扩展类方法虽能增加容量，但未定量评估可塑性，且常冻结旧参数导致利用率低。
- **内存增长问题**：GPM等方法存储的记忆基矩阵维度随任务数单调递增，内存开销持续增大。

## 核心贡献（创新点）
- **DualGPM**：通过维护梯度子空间 $\mathcal{M}_{l,t}$ 及其正交补 $\mathcal{M}_{l,t}^{\perp}$ 中较小者的正交基进行投影，使内存先增后减，相比GPM大幅节省存储。
- **AGRR可塑性度量**：首次定义平均梯度保留率（AGRR）量化模型在新任务上的参数更新潜力，并与最终准确率强相关。
- **自适应可塑性改进**：基于GRR评估结果自适应扩展输入维度（而非固定扩展），用更少参数获得更高可塑性。
- **统一框架**：DualGPM+AGRR+自适应扩展构成完整的"评估-改进"闭环，在Accuracy和Memory上均优于SOTA。

## 方法详解
**1. DualGPM（双梯度投影记忆）**
- 核心洞察：梯度更新位于输入张成的子空间 $\mathcal{M}_{l,t}$ 内（Proposition 1）。
- 投影公式：若存储 $M_{l,t}$（$\mathcal{M}_{l,t}$ 的正交基），则 $\hat{g}_{l,t} = g_{l,t} - M_{l,t}(M_{l,t})^T g_{l,t}$；若存储 $M_{l,t}^{\perp}$（正交补的正交基），则 $\hat{g}_{l,t} = M_{l,t}^{\perp}(M_{l,t}^{\perp})^T g_{l,t}$，两者等价。
- 切换策略：当 $\dim(\mathcal{M}_{l,t}) > \dim(\mathcal{M}_{l,t}^{\perp})$ 时通过SVD将基从 $\mathcal{M}_{l,t}$ 转为 $\mathcal{M}_{l,t}^{\perp}$，始终存储较少的基。
- 可扩展参数层处理：将已有基嵌入到更高维空间（补零），再进行子空间更新。

**2. 可塑性评估（AGRR）**
- 单层梯度保留率：$\mathrm{GRR}(l,t) = \mathbb{E}_{x \sim \mathcal{D}_t}\left[\frac{\|\hat{g}_{l,t}\|_2}{\|g_{l,t}\|_2}\right]$，值越小表示投影移除越多，可塑性越低。
- 平均梯度保留率：$\mathrm{AGRR}(t) = \frac{1}{L}\sum_{l=1}^{L}\mathrm{GRR}(l,t)$。
- 实验验证：AGRR与平均梯度范数和最终准确率均呈正相关。

**3. 自适应可塑性改进**
- 扩展维度决策：$d_I^{l,t} = d_I^{l,t-1} + \max\left(\left\lfloor K(\rho - \mathrm{GRR}(l,t)) + 0.5 \right\rfloor, 0\right)$，GRR越低则扩展越多。
- 输入扩展：新增变换 $\Phi_{l,t}(\mathbf{h}_l) = B_{l,t} \bullet h_l$，拼接后作为新输入。
- 训练约束：$B_{l,t}$ 中对应前 $t-1$ 任务的部分冻结，仅训练新任务部分。
- 推理时仅使用 $W_{l,t}$（不展开）。

## 实验与结果
**数据集**：Split CIFAR100（20任务×5类）、CIFAR100-sup（20任务）、Split Mini-Imagenet（20任务）、5-Datasets（5个异构数据集）。

**评估指标**：ACC（平均准确率）、BWT（负向迁移/遗忘）、内存占用。

**主要结果**（Table 1）：
- **CIFAR100-sup**：API 60.2% ACC / −0.2% BWT，超越GPM（57.7%/−1.2%）2.5个百分点。
- **Split CIFAR100**：API 81.4% ACC，超越GPM（78.9%）2.5个百分点；BWT −0.8%。
- **Split Mini-Imagenet**：API 65.9% ACC，超越GPM（61.2%）4.7个百分点。
- **5-Datasets**：API 91.1% ACC，超越GPM（88.8%）2.3个百分点。

**内存对比**（Table 3）：Split CIFAR100上API仅用2.0M（DualGPM+扩展），GPM需7.3M；5-Datasets上API 3.1M vs GPM 7.7M。

**扩展方法对比**（Table 2）：API容量仅105%（GPM为100%），远低于RKR（116%）、DEN（191%），但精度最优。

**消融**：用GPM替换DualGPM的API（API(GPM)）精度相近但内存多3.6×；等量扩展（Equal）需更多参数才能达到同等精度。

**结论**：API在所有数据集上均取得SOTA精度，同时内存开销最低。

## 相关工作脉络
- **GPM（Gradient Projection Memory）**：API的直接前身，仅维护 $\mathcal{M}_{l,t}$ 基，内存单调增长；DualGPM通过正交补视角解决了这一缺陷。
- **EWC（Elastic Weight Consolidation）**：正则化类代表，通过Fisher信息评估参数重要性施加惩罚；缺点是无法避免可塑性随任务累积下降。
- **GEM/A-GEM**：经验回放类，保存旧样本估计梯度；与API的本质区别是依赖真实样本（隐私问题），API完全不存储样本。
- **TRGP（Trust Region Gradient Projection）**：同属投影类，定义信任区域改善新任务性能；API进一步引入可塑性度量与自适应扩展。
- **PNN/DEN/APD/RKR（扩展类）**：均通过增加参数提升容量，但未量化评估可塑性需求，且常冻结旧参数；API的扩展是"按需、最小化"的。
- **Connector**：通过线性连接器改善可塑性-稳定性权衡，但未建立可塑性的显式度量。

## 局限性与未来方向
- 仅在任务增量设置（task-incremental，已知任务身份）下验证，泛化到无任务身份设置需进一步研究。
- 超参数 $\rho$ 和 $K$ 需手动调节，不同数据集/架构的最优值不同。
- 深层网络中频繁SVD操作可能带来额外计算开销（论文声明更新仅在任务间进行，影响有限）。
- 扩展策略仅针对输入维度，对卷积核维度的自适应扩展未涉及。
- 论文建议未来工作：扩展到无任务身份的设置、结合其他持续学习设定。

## 研究启发与可借鉴点
- **DualGPM的"双存储"思想**：通过维护较小空间的正交基来降低内存，这一策略可迁移到其他基于投影/子空间的持续学习方法中。
- **AGRR作为可塑性代理指标**：为后续研究提供可量化评估模型可塑性（而非仅拟合度）的通用度量，可借鉴到模型压缩、神经架构搜索等场景。
- **"评估-自适应改进"闭环范式**：先量化评估某一属性，再按需改进，而非盲目扩展；这一设计模式可用于其他AI系统的自适应调节。
- **冻结旧参数部分的轻量扩展**：扩展时只训练新任务部分，推理时不展开，兼顾效率与性能；对动态网络架构设计有参考价值。
- **与GPM的对比实验设计**：通过替换DualGPM为GPM的消融清晰证明改进有效性，实验设计严谨值得效仿。

## 关键术语表
- **Continual Learning（持续学习）**：模型按顺序学习多个任务，同时保持对旧任务性能的学习范式。
- **Catastrophic Forgetting（灾难性遗忘）**：神经网络在学习新任务时剧烈丢失已有知识的现象。
- **Stability-Plasticity Dilemma（稳定性-可塑性困境）**：持续学习中保持旧知识（稳定性）与学习新知识（可塑性）之间的内在矛盾。
- **Gradient Projection Memory（GPM）**：通过正交投影将新任务梯度投影到旧任务梯度子空间的正交补上，防止干扰旧知识。
- **Gradient Retention Ratio（GRR）**：投影后梯度范数与原梯度范数之比，衡量约束强度与可塑性水平。
- **Average GRR（AGRR）**：所有网络层GRR的平均值，作为模型整体可塑性的量化指标。
- **Task-incremental Setting（任务增量设置）**：推理阶段已知当前测试样本所属任务身份的实验设定。
- **BWT（Backward Transfer）**：负向迁移指标，衡量学习新任务后对旧任务性能的下降程度。

## 可复现要素
- **数据集**：Split CIFAR100、CIFAR100-sup、Split Mini-Imagenet、5-Datasets，均为公开数据集。
- **代码**：论文未明确提及代码开源状态（注：CVPR 2023通常要求开源，建议查阅官方GitHub）。
- **模型架构**：Split CIFAR100使用5层AlexNet；CIFAR100-sup使用修改版LeNet；Split Mini-Imagenet和5-Datasets使用简化版ResNet18。
- **训练细节**：SGD优化；Split CIFAR100训200轮，CIFAR100-sup训50轮，Split Mini-Imagenet训10轮，5-Datasets训100轮；batch size=64；早停策略；4×NVIDIA TITAN Xp GPU。
- **关键超参数**：$\epsilon_{th}^l$ 与GPM保持一致；$\rho=0.5$，$K=10$（默认值）；阈值 $\epsilon_{th}^l$ 控制子空间维度增长。
