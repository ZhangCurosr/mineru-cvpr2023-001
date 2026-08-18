---
title: "AttriCLIP-A-Non-Incremental-Learner-for-Incremental-Knowledg"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_AttriCLIP_A_Non-Incremental_Learner_for_Incremental_Knowledge_Learning_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:40:16"
---

# 论文速读：AttriCLIP-A-Non-Incremental-Learner-for-Incremental-Knowledg

## 一句话总结
提出 Attris... wait, AttriCLIP：一种基于冻结 CLIP 双塔与动态属性词库的非增量持续学习方法，通过按图像属性匹配并生成文本提示进行图文对比分类，在参数量固定且无需重放记忆的条件下实现高效、可泛化的知识增量。

## 研究问题与动机
1. 传统持续学习采用共享特征提取器+逐任务扩展分类器的架构，导致模型可训练参数随任务累积不断增长，难以适应类别无限开放的现实场景。
2. 分类器增量会引入严重的分类器偏差，现有方法通常依赖历史样本重放记忆（Replay Memory）来缓解灾难性遗忘，在存储受限或隐私敏感场景下不可行。
3. 现有 Prompt Tuning 持续学习方法（如 L2P、DualPrompt）仅将可学习提示附加在视觉特征端，未充分挖掘多模态 CLIP 中文本侧的语义对齐与泛化潜力。
4. 现有评测多为理想化的单数据集顺序切分，缺乏对长序列域偏移（Domain-Shift）与跨数据集知识迁移的系统性评估。

## 核心贡献（创新点）
1. **属性驱动的动态提示选择机制**：摒弃按类别/任务硬绑定提示的传统做法，改为根据图像局部属性从词库中动态选取提示，从根本上避免同一模型顺序微调导致的知识覆盖。
2. **非增量架构设计**：冻结 CLIP 图像与文本编码器，仅训练固定规模的属性词库（Key-Prompt 对），参数量不随任务数增长，且完全无需重放缓冲区。
3. **图文协同的对比分类范式**：将选中的属性提示与类别名拼接构成动态文本描述，通过图文余弦相似度完成分类，使模型以“语义属性对齐”替代“类别权重扩张”。
4. **提出 CDCL 评估设定与迁移指标**：构建 Cross-Datasets Continual Learning 长序列域偏移评测框架，并引入前向迁移（FT）与后向迁移（BT）量化知识巩固与泛化能力。

## 方法详解
- **基础组件**：冻结预训练的 CLIP 图像编码器 $f_\theta(\cdot)$ 与文本编码器 $g_\psi(\cdot)$。设计属性词库 $\{ \mathcal{K}, \mathcal{P} \} = \{ (\mathbf{k}_1, \mathbf{P}_1), \ldots, (\mathbf{k}_N, \mathbf{P}_N) \}$，其中 $\mathbf{k}_i \in \mathbb{R}^D$ 为视觉属性键，$\mathbf{P}_i \in \mathbb{R}^{D \times M}$ 为文本提示
