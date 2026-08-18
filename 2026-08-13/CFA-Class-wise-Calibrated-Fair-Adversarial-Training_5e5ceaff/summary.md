---
title: "CFA-Class-wise-Calibrated-Fair-Adversarial-Training"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_CFA_Class-Wise_Calibrated_Fair_Adversarial_Training_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:40:57"
field: "对抗鲁棒性与公平性"
keywords: ["adversarial training", "robustness fairness", "class-wise calibration", "automated machine learning", "deep learning security"]
innovations: ["首次系统揭示不同类别对对抗配置的差异化偏好并提供理论证明", "提出CFA框架实现类别级自适应训练配置（CCM+CCR+FAWA）", "FAWA通过checkpoint阈值过滤稳定最坏类别鲁棒性提升"]
benchmarks: ["CIFAR-10", "Tiny-ImageNet"]
---

# 论文速读：CFA-Class-wise-Calibrated-Fair-Adversarial-Training

## 一句话总结
本文首次从理论上和实证上研究了不同类别对对抗配置（扰动边界、正则化、权重平均）的偏好差异，并提出 CFA（Class-wise Calibrated Fair Adversarial Training）框架，通过为每个类别自适应定制训练配置，在提升整体对抗鲁棒性的同时显著改善类别级鲁棒性公平性。

## 研究问题与动机
- **核心问题**：现有对抗训练方法普遍关注整体鲁棒性，但模型在不同类别上的鲁棒性存在显著差异（鲁棒性不公平），例如自动驾驶中"停止标志"类别可能极易被攻击，而整体指标看起来良好。
- **理论发现**：强对抗攻击对"困难类别"（clean accuracy较低的类别）有害——随着攻击强度增加，困难类别的clean accuracy下降更快，而robust accuracy提升更慢。
- **训练波动现象**：最坏类别鲁棒性在训练过程中剧烈波动（相邻epoch间可达10%差异），仅选择整体鲁棒性最佳的checkpoint会导致极差的鲁棒公平性。
- **现有方法不足**：FRL等方法通过增大margin改进公平性，但会牺牲整体鲁棒性峰值；instance-wise方法仅关注整体鲁棒性，对公平性无帮助。

## 核心贡献（创新点）
- **理论洞察**：首次系统揭示不同类别对对抗配置（扰动margin、正则化、权重平均）存在差异化偏好，并提供理论证明与实证验证。
- **CFA框架**：提出包含三个组件的训练框架——CCM（类别自适应扰动边界）、CCR（类别自适应正则化）、FAWA（公平感知权重平均），自动为每类定制训练配置。
- **FAWA机制**：在EMA权重平均中引入阈值过滤，仅采纳最坏类别鲁棒性高于阈值的checkpoint，稳定并提升最坏类别表现。
- **SOTA性能**：在CIFAR-10上，TRADES+CFA的最佳checkpoint平均鲁棒准确率50.1%，最坏类别鲁棒准确率26.5%，显著优于FRL等方法，且可无缝集成至现有AT方法。

## 方法详解
CFA框架由三个核心组件构成：

**1. CCM（Class-wise Calibrated Margin）类别自适应扰动边界**
- 利用上一epoch各分类的训练鲁棒准确率 $t_k$ 作为难度度量
- 为第k类自适应调整扰动边界：$\epsilon_k \leftarrow (\lambda_1 + t_k) \cdot \epsilon$
- 困难类别（$t_k$小）获得较小margin，避免过度攻击导致clean accuracy大幅损失

**2. CCR（Class-wise Calibrated Regularization）类别自适应正则化**
- 专为TRADES设计，调整robustness regularization系数：$\beta_k \leftarrow (\lambda_2 + t_k) \cdot \beta$
- 修正后的目标函数：$\mathcal{L}_{\pmb{\theta}}(\beta; x, y) = \frac{\mathcal{L}(\pmb{\theta}; x, y) + \beta_y \max_{\|x'-x\|\leq\epsilon} K(f_\theta(x), f_\theta(x'))}{1+\beta_y}$
- 分母$(1+\beta_y)$平衡各类别权重，困难类别获得更高的natural loss权重

**3. FAWA（Fairness Aware Weight Averaging）公平感知权重平均**
- 在EMA过程中设置阈值$\delta$，仅当checkpoint的最坏类别鲁棒性 $\geq \delta$ 时才纳入平均
- 消除"不公平"checkpoint，显著降低最坏类别鲁棒性的波动
- 从validation集（2%样本）评估checkpoint公平性，不增加额外训练成本

## 实验与结果
- **数据集**：CIFAR-10（主实验），Tiny-ImageNet（附录）
- **模型**：PreActResNet-18
- **评估指标**：AutoAttack（AA）鲁棒准确率，分别报告平均值和最坏类别值；clean准确率
- **基线**：Vanilla AT、TRADES、FAT、FRL，以及EMA变体
- **核心结果（Table 1，最佳checkpoint）**：
  - TRADES+CFA：Clean 80.4%，Avg AA 50.1%，Worst AA 26.5%
  - 相比TRADES：Avg提升1.8%，Worst提升4.8%
  - 相比FRL：Avg提升约4%，Worst提升约1%
  - AT+CFA：Worst AA达24.4%，比AT+EMA高3.1%
- **消融实验**：CCM、CCR、FAWA各自独立有效；$\lambda_1=0.5$为最优选择；困难类别使用更小margin，易类别使用更大margin，符合理论预期

## 相关工作脉络
- **FRL [29]**：唯一专注于对抗鲁棒性公平的工作，通过remargin和reweight调整，但会牺牲整体鲁棒性峰值，且不考虑训练波动问题。
- **Instance-wise自适应AT [3,7,10,26,31]**：如FAT为每个样本自适应调整攻击强度，但仅关注整体鲁棒性，对公平性无帮助（FAT worst AA仅17.2%）。
- **Theoretical robustness [22]**：Tsipras等人的工作揭示了鲁棒性与准确率的trade-off，本文在此基础上进一步分析类别差异。
- **Robust overfitting [19]**：Rice等人的工作指出对抗训练后期出现过拟合，本文通过FAWA缓解该问题。
- **Class-wise robustness分析 [4,21,29]**：Benz、Tian等人的工作揭示了类别鲁棒性差异现象，本文首次系统性提出解决方案。

## 局限性与未来方向
- **类别粒度限制**：当前方法在类别级别进行校准，未探索instance级别更细粒度的自适应，可能丢失类别内样本的异质性信息。
- **阈值依赖**：FAWA需要手动设置公平性阈值$\delta$，缺乏自动调优机制。
- **扩展性待验证**：仅在CIFAR-10和Tiny-ImageNet上验证，对更大规模数据集（如ImageNet）和复杂架构（如ViT）的效果未知。
- **计算开销**：虽然FAWA不增加训练成本，但需要额外的validation评估步骤。

## 研究启发与可借鉴点
- **类别难度度量**：使用训练鲁棒准确率作为类别难度信号的方法简洁有效，可迁移至其他公平性研究场景。
- **配置自适应框架**：CCM/CCR的自适应调度机制设计思想可复用到其他超参数（如学习率、batch size）的类别级自适应优化。
- **checkpoint筛选策略**：FAWA的阈值过滤思想可用于其他需要稳定模型质量的训练场景，避免"幸存者偏差"。
- **理论与实践结合**：从理论分析（Theorem 1-4）出发指导实验设计的方法论值得借鉴。
- **可插拔设计**：CFA各组件可独立或组合使用，与现有方法兼容性强，便于快速验证和提升。

## 关键术语表
**Adversarial Training (AT)**：通过min-max优化训练模型，内层最大化攻击损失，外层最小化攻击后的损失，是最有效的对抗鲁棒性提升方法。
**Class-wise Robustness Fairness**：指模型在不同类别上的对抗鲁棒性差异程度，差异越小表示公平性越好。
**Fairness Aware Weight Averaging (FAWA)**：在EMA权重平均中引入公平性阈值过滤，仅采纳最坏类别鲁棒性达标的checkpoint。
**Customized Class-wise perturbation Margin (CCM)**：根据每类训练鲁棒准确率自适应调整扰动边界，困难类别使用较小margin。
**Customized Class-wise Regularization (CCR)**：为TRADES设计，按类别自适应调整正则化系数$\beta$，平衡clean和robust损失权重。
**AutoAttack (AA)**：集成多种参数-free攻击的鲁棒性评估基准，相比单一PGD攻击更可靠。
**Robust Overfitting**：对抗训练后期出现过拟合现象，模型在训练集上鲁棒性提升但测试集鲁棒性下降。
**Instance-wise Adaptive AT**：为每个训练样本自适应调整攻击强度的方法，与本文的类别级方法形成对比。

## 可复现要素
- **数据集**：CIFAR-10（公开），Tiny-ImageNet（公开）
- **代码**：开源，GitHub地址 https://github.com/PKU-ML/CFA
- **模型**：PreActResNet-18
- **关键超参**：
  - 初始扰动边界 $\epsilon = 8/255$
  - TRADES初始正则化 $\beta = 6$
  - CCM基线 $\lambda_1 = 0.5$（AT）/ $0.3$（TRADES）
  - CCR $\lambda_2 = 0.5$
  - EMA/FAWA decay rate = 0.85，从epoch 50开始
  - FAWA公平性阈值 $\delta = 0.2$
  - 2% validation样本用于FAWA评估
