---
title: "CFA-Class-wise-Calibrated-Fair-Adversarial-Training"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Wei_CFA_Class-Wise_Calibrated_Fair_Adversarial_Training_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:40:42"
field: "对抗鲁棒性与公平性"
keywords: ["adversarial training", "robust fairness", "class-wise robustness", "fair adversarial training", "customized training configuration"]
innovations: ["首次从类别视角提出自适应对抗训练框架CFA，动态定制每类别的扰动边界和正则化", "提出FAWA机制，在EMA中过滤不公平checkpoint以稳定最坏类别鲁棒性", "理论证明强攻击对困难类别有害，揭示类别鲁棒差异的内在原因"]
benchmarks: ["CIFAR-10", "Tiny-ImageNet"]
---

# 论文速读：CFA-Class-wise-Calibrated-Fair-Adversarial-Training

## 一句话总结
本文首次从类别视角研究对抗训练中的鲁棒公平性问题，提出CFA（Class-wise calibrated Fair Adversarial training）框架，通过为每个类别动态定制扰动边界、正则化权重和公平感知的权重平均策略，在不牺牲整体鲁棒性的同时显著提升类别间鲁棒公平性。

## 研究问题与动机
- **类别鲁棒性差异普遍存在**：对抗训练模型在某些类别上鲁棒性强，而在另一些类别上极其脆弱（如自动驾驶中的"停止标志"），引发安全性隐患。
- **现有公平方法存在局限**：FRL等方法在提升公平性时以牺牲整体鲁棒性为代价，且仅通过盲目增大扰动边界来缓解困难类别的过拟合，无法达到最优性能。
- **强攻击对困难类别有害**：理论分析揭示，对于内在困难类别（clean accuracy较低的类别），更强的对抗攻击反而导致更大的clean accuracy损失，而robust accuracy增益有限。
- **最坏类别鲁棒性剧烈波动**：训练过程中最坏类别的鲁棒性在相邻checkpoint间可能波动近10%，仅选择整体鲁棒性最佳的checkpoint可能导致极不公平的模型。

## 核心贡献（创新点）
- **理论揭示类别差异的本质原因**：通过二元分类任务的理论分析，证明鲁棒特征可靠性差异是导致类别间困难度差异的根本原因，并证明强攻击对困难类别有害。
- **提出类别级自适应对抗训练框架CFA**：首次从类别视角而非实例视角设计自适应配置，与现有instance-wise方法在目标（公平性）和粒度（类别级更稳定）上本质不同。
- **设计公平感知权重平均（FAWA）**：在EMA过程中引入公平性阈值过滤掉不公平的checkpoint，稳定最坏类别的鲁棒性，这是现有方法未考虑的关键机制。
- **兼具整体鲁棒性与类别公平性**：与FRL不同，CFA不牺牲整体鲁棒性即可提升最坏类别的鲁棒性，在CIFAR-10上将最坏类别AA鲁棒性提升约4%（TRADES+CFA vs TRADES）。

## 方法详解
**整体框架**：CFA由三个核心组件构成：CCM（类别自适应扰动边界）、CCR（类别自适应正则化）、FAWA（公平感知权重平均）。

**CCM（Class-wise Calibrated Margin）**：基于上一epoch各类别的训练鲁棒准确率 $t_k$ 动态调整扰动边界：
$$\epsilon_k \leftarrow (\lambda_1 + t_k) \cdot \epsilon$$
其中 $\lambda_1$ 为基础扰动预算（AT设为0.5，TRADES设为0.3）。困难类别（$t_k$ 低）获得更小的 $\epsilon_k$，避免过度损害；易类别获得更大的 $\epsilon_k$，进一步提升鲁棒性。该配置可自适应收敛至合适范围。

**CCR（Class-wise Calibrated Regularization）**：针对TRADES框架，为每类别定制正则化系数：
$$\beta_y \leftarrow (\lambda_2 + t_y) \cdot \beta$$
修改后的损失函数为：
$$\mathcal{L}_{\pmb{\theta}}(\beta; x, y) = \frac{\mathcal{L}(\pmb{\theta}; x, y) + \beta_y \max_{\|x'-x\|\leq\epsilon} K(f_{\pmb{\theta}}(x), f_{\pmb{\theta}}(x'))}{1+\beta_y}$$
分母 $1+\beta_y$ 确保不同类别间的权重平衡，困难类别获得更高的自然损失权重。

**FAWA（Fairness Aware Weight Averaging）**：在EMA过程中，仅当checkpoint的最坏类别鲁棒性超过阈值 $\delta$ 时才纳入平均：
```
if min_{y∈Y} R_y(f_θ, D_val) ≥ δ then
    θ̄ ← α·θ̄ + (1-α)·θ
```
从数据集中抽取2%样本作为验证集（不增加额外计算成本），公平阈值 $\delta=0.2$，衰减率 $\alpha=0.85$，从第50epoch开始。

## 实验与结果
- **数据集**：CIFAR-10（主实验），Tiny-ImageNet（附录），使用PreActResNet-18模型。
- **评估指标**：Clean Accuracy、AA鲁棒准确率（AutoAttack），均报告Average和Worst（最坏类别），分别在best和last checkpoint。
- **基线方法**：AT、TRADES、FAT、FRL及其EMA变体。
- **核心结果**（TRADES+CFA，best checkpoint）：
  - Average AA鲁棒准确率：**50.1%**（vs TRADES的48.3%，提升1.8%）
  - Worst AA鲁棒准确率：**26.5%**（vs TRADES的21.7%，提升**4.8%**）
  - 相比FRL+EMA，average提升约4%，worst提升约1%。
- **消融实验**：CCM单独使用即可提升worst鲁棒性（AT: 20.1→22.8）；CCR进一步提升TRADES性能；FAWA相比EMA将worst鲁棒性再提升约2%（AT: 21.3→23.1）。
- **λ₁敏感性**：λ₁在0.3~0.7范围内CFA均优于vanilla AT，λ₁=0.5表现最佳。

## 相关工作脉络
- **Fair Robust Learning (FRL)**：唯一专门针对类别鲁棒公平性的对抗训练方法，通过remargin和reweight在约束违反时微调；但仅缓解过拟合无法达到最优，且以牺牲整体性能为代价。CFA通过类别级自适应配置实现两者兼顾。
- **Instance-wise Adaptive AT (FAT等)**：为每个样本定制对抗配置以提升整体鲁棒性；但实例级方法对公平性无帮助（FAT的worst仅17.2%），且波动频繁。CFA从类别视角出发，在灵活性和稳定性间取得更好平衡。
- **Exponential Moving Average (EMA)**：经典的模型权重平滑技术；FAWA是其公平性增强版本，通过阈值过滤不公平checkpoint。
- **Robustness-Accuracy Tradeoff理论**：Tsipras et al. (2018) 证明鲁棒性与accuracy的对立关系；本文在此基础上进一步分析类别层面的tradeoff差异。
- **Curriculum/Customized AT**：CAT、MMA等自定义对抗训练方法；本文强调首次从类别公平性而非单纯整体鲁棒性角度设计自适应机制。

## 局限性与未来方向
- 理论分析基于简化的SVM二元分类toy model，实际DNN的类别差异机制更为复杂。
- 仅在CIFAR-10和Tiny-ImageNet上验证，在更复杂数据集（如ImageNet）或未公开实验中未验证。
- FAWA依赖验证集评估最坏类别鲁棒性，虽仅占2%但需额外前向计算。
- 未讨论与其他instance-wise自适应方法的融合潜力（文中提及可结合但未实验）。
- 超参数（λ₁、λ₂、δ）需根据具体任务调整，泛化性有待进一步验证。

## 研究启发与可借鉴点
- **类别级自适应设计范式**：将自适应机制从instance级提升到class级，在灵活性和稳定性间取得更优平衡，该思路可迁移至其他 fairness-aware 的ML训练场景。
- **训练过程监控指标的创新**：发现"最坏类别鲁棒性波动远大于整体鲁棒性"这一现象，提示在鲁棒训练中应同时监控类别级指标而非仅看整体指标。
- **权重平均的公平性增强**：FAWA的checkpoint过滤策略简单有效，可推广到其他基于模型集成/平均的鲁棒训练方法中。
- **理论-实验闭环**：从toy model理论分析→ empirical验证→ 方法设计，这种研究路径值得借鉴，尤其是用参数w隐式表征attack strength的技巧。
- **与团队方向结合机会**：若团队关注医学图像分析等安全关键场景中的鲁棒公平性，CFA的类别级校准机制可直接应用于解决不同疾病类别间模型表现差异问题。

## 关键术语表
- **Adversarial Training (AT)**：通过min-max优化训练模型，内层最大化扰动损失、外层最小化，以提升模型对对抗扰动的鲁棒性。
- **Robust Fairness**：指模型在不同类别间鲁棒性能的一致性，即最坏类别的鲁棒性不应显著低于整体水平。
- **Class-wise Calibrated Margin (CCM)**：根据每类别上一epoch的训练鲁棒准确率动态调整对抗扰动边界，困难类别使用更小边界。
- **Class-wise Calibrated Regularization (CCR)**：为TRADES中每类别定制正则化系数，困难类别降低正则化权重以保护clean accuracy。
- **Fairness Aware Weight Averaging (FAWA)**：在EMA过程中仅纳入最坏类别鲁棒性超过阈值的checkpoint，过滤不公平的中间模型。
- **AutoAttack (AA)**：集成多种参数-free对抗攻击的鲁棒性评估标准，比单一PGD攻击更可靠。
- **Hard Class vs Easy Class**：hard class指clean accuracy较低的类别，其鲁棒特征更不可靠，对强攻击更敏感。
- **Overfitting in Adversarial Training**：训练后期整体鲁棒性不再提升甚至下降，但最坏类别可能因过拟合而表现异常波动。

## 可复现要素
- **数据集**：CIFAR-10（公开）、Tiny-ImageNet（公开）
- **代码**：已开源，https://github.com/PKU-ML/CFA
- **模型**：PreActResNet-18
- **关键超参**：
  - 扰动边界 ε = 8/255（CIFAR-10）
  - TRADES正则化 β = 6
  - CCM基线 λ₁ = 0.5（AT）、0.3（TRADES）
  - CCR正则化 λ₂ = 0.5
  - FAWA阈值 δ = 0.2，衰减率 α = 0.85
  - EMA/FAWA从第50epoch开始
  - 验证集占比2%
- **训练设置**：SGD momentum=0.9，weight decay=5×10⁻⁴，initial lr=0.1，200 epochs，lr在100和150epoch各除以10
