---
title: "Class-Conditional-Sharpness-Aware-Minimization-for-Deep-Long"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Zhou_Class-Conditional_Sharpness-Aware_Minimization_for_Deep_Long-Tailed_Recognition_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:42:41"
field: "长尾视觉识别"
keywords: ["long-tailed recognition", "sharpness-aware minimization", "PAC-Bayesian", "class-conditional", "decoupled training", "robust optimization"]
innovations: ["基于PAC-Bayesian理论推导类条件扰动半径，实现类自适应的平坦化优化", "将CC-SAM集成到两阶段解耦框架，第一阶段类条件参数扰动+第二阶段对抗特征训练", "在多个长尾benchmark上取得SOTA或接近SOTA性能，并在开放长尾识别中展现强鲁棒性"]
benchmarks: ["CIFAR-10-LT", "CIFAR-100-LT", "Places-LT", "ImageNet-LT", "iNaturalist 2018"]
---

# 论文速读：Class-Conditional Sharpness-Aware Minimization for Deep Long-Tailed Recognition

## 一句话总结
本文提出 CC-SAM（Class-Conditional Sharpness-Aware Minimization），通过在两阶段解耦训练框架中引入类条件参数的平滑优化，缓解深度长尾识别中普遍存在的尖锐极小值问题，在多个主流长尾 benchmark 上达到竞争性性能。

## 研究问题与动机
- 深度学习模型平坦极小值通常对应更好泛化，但在深度长尾识别（DLTR）场景下，这一问题尚未被深入探索。
- 现有 sharpness-aware 方法（如 SAM）直接应用于长尾学习时效果有限，因为严重标签分布偏移（label distribution shift）导致全局扰动策略失效。
- 直觉的平坦化操作（如 SWA、SN、GP、MP）集成到长尾方法后几乎无增益甚至负收益，说明需要专门针对长尾分布设计的平坦化方案。
- 通过 PAC-Bayesian 理论推导发现，最优扰动半径应依赖于各类样本量，多数类需要更小扰动区域，少类需要更大扰动——这揭示了类条件平坦化的必要性。

## 核心贡献（创新点）
- 从平坦化视角系统验证了现有 sharpness-aware 方法在长尾学习中的次优性，明确了需要专门设计的方向。
- 提出 CC-SAM，基于 PAC-Bayesian 框架推导类条件扰动半径，实现高效的类条件参数平滑训练。
- 将 CC-SAM 与两阶段解耦训练范式结合，在第二阶段利用类平衡采样和对抗特征生成进一步鲁棒化分类器。
- 在 CIFAR-LT、Places-LT、ImageNet-LT、iNaturalist 2018 等多个 benchmark 上取得有竞争力的结果，并在开放长尾识别任务中展现出色鲁棒性。

## 方法详解
- **两阶段解耦框架**：遵循 DLTR 的 decoupling 范式，第一阶段同时训练特征提取器 f(x;φ) 和分类器 h(z;θ)，第二阶段冻结 φ 专注优化分类器。
- **第一阶段 - 类条件 SAM（CC-SAM）**：基于 PAC-Bayesian 理论推导各类的特征平坦半径 ρ_c* = (||w||_2 / (2||∇_w L_S^c(w)||_2))^{1/2} · k^{-1/4} · (n_c - 1)^{-1/4}，其中 ρ_c* 与类别样本数 n_c 负相关——多数类用较小扰动，少类用较大扰动。扰动向量 ε_c* = √k · ρ_c* · ∇_w L_S^c(w) / ||∇_w L_S^c(w)||_2，通过一阶近似高效计算，无需高阶项。
- **第二阶段 - 分类器对抗训练**：冻结 backbone，采用类平衡采样（class-balanced sampling），在前向传播中生成对抗特征 z_adv = z + λ · ∇_z L_S(z;θ) / ||∇_z L_S(z;θ)||_2，并采用渐进策略 L = (1 - t/T)·L_S + (t/T)·L_A 使对抗损失逐渐主导训练。

## 实验与结果
- **数据集**：CIFAR-10-LT/CIFAR-100-LT（不平衡比 200/100/50）、Places-LT（IR=996）、ImageNet-LT（IR=256）、iNaturalist 2018（IR=512）。
- **主要结果**：CIFAR-10-LT (IR=200) 达 80.94%，CIFAR-100-LT (IR=100) 达 50.83%，Places-LT 达 40.6%，ImageNet-LT (ResNeXt-50) 整体 55.4%（few classes: 41.1%），iNaturalist 2018 整体 70.9%。
- **最强提升**：在 CIFAR-LT 系列上统一最优；ImageNet-LT 上 few class 达到 41.1%，显著优于 GCL 等基线（35.5% vs 41.1% 提升约 5.6pp）。
- **开放长尾识别**：在 ImageNet-LT 和 Places-LT 上均取得最优 F-measure（0.552 和 0.510）。

## 相关工作脉络
- **BGP [49]**：基于 logits 与梯度范数相关性提出梯度惩罚正则，但缺乏严格理论解释且使用过松的 PAC-Bayesian bound，性能低于 CC-SAM。
- **VS+SAM [38]**：直接使用 SAM 缓解长尾重加权导致的 saddle points，但未指定最优扰动尺度或提供理论依据；CC-SAM 在此基础上提供类条件设计并给出理论保证。
- **Liu et al. [30]**：提出 instance-level 重加权 SAM 应对数据不均衡，但未给出最优扰动推导；作者认为这是 CC-SAM 的简化不完整版本。
- **Decoupled Training [20]**：本文采用的两阶段框架基础，CC-SAM 在其第一阶段引入类条件平坦化。
- **MiSLAS [62]、GCL [26]**：近期 SOTA 方法，CC-SAM 在其基础上进一步提升少类性能。

## 局限性与未来方向
- 论文提到计算开销增加（额外梯度下降），仅扰动最后几层作为高效版本；完整全层扰动效率待优化。
- 类条件扰动假设每类独立估计梯度，极端长尾（极少样本类）的梯度估计可能不稳定。
- 未深入探讨与其他类别重平衡策略（如 Balanced Softmax）的兼容性融合。
- 未来可扩展至更多分布外泛化场景（如域适应、持续学习）。

## 研究启发与可借鉴点
- **PAC-Bayesian 理论驱动设计**：通过理论推导确定超参数（扰动半径）的类条件依赖关系，为方法论创新提供坚实理论基础，值得借鉴于其他不均衡学习问题。
- **两阶段解耦 + 对抗训练结合**：Stage 2 的对抗特征生成采用渐进式权重策略 (1-t/T)·L_S + (t/T)·L_A，可迁移至其他需要分类器鲁棒性的场景。
- **高效近似避免高阶计算**：利用一阶泰勒展开近似扰动向量的技巧，在接近最优解时梯度趋于零但类级梯度非零，避免了二阶计算，实用价值高。
- **类平衡采样 + 特征空间扰动**：在特征空间而非参数空间施加对抗扰动，对 backbone 冻结阶段的 classifier refinement 有启发意义。

## 关键术语表
**Deep Long-Tailed Recognition (DLTR)**：训练数据标签分布极度不均衡（多数类样本远多于少数类），但测试期望均匀评估的视觉分类问题。
**Sharpness-Aware Minimization (SAM)**：一种寻找损失景观中平坦极小值的优化策略，通过在参数邻域内最大化损失后再更新，提升模型泛化能力。
**PAC-Bayesian Framework**：基于概率论的泛化误差分析框架，通过先验/后验分布的 KL 散度给出泛化误差上界。
**Decoupling Paradigm**：将特征提取器和分类器分开训练的两阶段范式，第一阶段用原始分布训练特征，第二阶段用类平衡采样训练分类器。
**Class-Conditional**：按类别分别处理，多数类和少数类采用不同的训练策略或超参数设置。
**Open Long-Tailed Recognition (OLTR)**：结合开放集识别与长尾分布的分类任务，模型需同时处理已知类和未知 OOD 样本。

## 可复现要素
- **数据集**：CIFAR-10-LT、CIFAR-100-LT、Places-LT、ImageNet-LT、iNaturalist 2018（均公开可获取）
- **代码**：已开源 https://github.com/zzpustc/CC-SAM
- **关键超参**：batch size 64/128/256/512（依数据集而定），SGD optimizer with momentum 0.9，λ（对抗特征步长），T（第二阶段总 epoch 数）
- **实现细节**：PyTorch 1.4.0，Tesla V100 GPU，仅扰动最后几层作为高效版本（论文未提及具体层数细节）
