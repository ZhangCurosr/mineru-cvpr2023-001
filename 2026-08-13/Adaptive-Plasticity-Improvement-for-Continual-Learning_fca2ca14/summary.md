---
title: "Adaptive-Plasticity-Improvement-for-Continual-Learning"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Liang_Adaptive_Plasticity_Improvement_for_Continual_Learning_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:50:01"
---

# 论文速读：Adaptive-Plasticity-Improvement-for-Continual-Learning

## 一句话总结
本文提出自适应塑性提升（API）方法，在克服灾难性遗忘的同时，首次引入梯度保留率（AGRR）定量评估模型塑性，并据此按需自适应扩展网络输入维度以提升新任务学习能力，在四个主流基准上以更低内存占用取得最高准确率。

## 研究问题与动机
1. **稳定性挤压塑性的固有矛盾**：现有正则化与记忆类方法为防止旧知识退化，会随任务数增加不断收紧参数约束，导致模型对新任务的适应性（塑性）持续下降。
2. **梯度投影方法内存单调增长**：以 GPM 为代表的梯度投影方法需维护旧任务梯度张成的子空间基，基维度随任务增加而单调递增，造成存储开销不可控。
3. **扩展类方法缺乏塑性评估与按需机制**：DEN、APD、CCLL 等虽通过增加参数提升容量，但通常等量扩张或冻结旧层，未考虑当前模型是否真正需要扩展，也未对塑性进行定量度量。

## 核心贡献（创新点）
1. **提出 DualGPM 抗遗忘机制**：动态维护旧任务梯度子空间 $\mathcal{M}_{l,t}$ 或其正交补 $\mathcal{M}_{l,t}^{\perp}$ 中维数较小的一组正交基，使内存占用维持在 $\min\{\dim(\mathcal{M}), \dim(\mathcal{M}^\perp)\}$，本质区别在于 GPM 仅维护 $\mathcal{M}_{l,t}$ 导致内存持续增长，而 DualGPM 通过对偶切换实现内存先升后降。
