---
title: "Burstormer-Burst-Image-Restoration-and-Enhancement-Transform"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Dudhane_Burstormer_Burst_Image_Restoration_and_Enhancement_Transformer_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:40:54"
---

# 论文速读：Burstormer-Burst-Image-Restoration-and-Enhancement-Transform

## 一句话总结
本文提出了 Burstormer，一种基于 Transformer 的突发图像恢复与增强统一框架，通过多尺度局部与非局部特征对齐及自适应帧间融合，有效解决了手持拍摄中因相机抖动/物体运动导致的亚像素错位与复杂退化问题，在突发超分、低光增强和去噪任务上均达到 SOTA。

## 研究问题与动机
- 现代智能手机受限于物理空间，单帧成像常存在分辨率低、动态范围窄、噪声大及低光色彩失真等问题，亟需利用多帧突发（burst）拍摄的互补信息提升画质。
- 现有方法在处理突发帧时面临两大瓶颈：一是帧间不可避免的亚像素偏移难以被单级显式对齐或可变形卷积充分校正，快速运动场景易产生模糊与鬼影；二是多帧特征聚合策略僵化（如固定帧数刚性融合或晚期融合），限制了灵活的帧间通信与计算效率。
- 隐式对齐代表工作 BIPNet 虽避免了显式光流估计，但缺乏多尺度全局上下文交互，且在复杂位移残留误差的修正上能力有限。
- 因此，研究需要一种能统一建模局部与非局部上下文、支持多尺度层次对齐、并具备灵活轻量级帧间通信能力的端到端架构。

## 核心贡献（创新点）
1. **提出 Burstormer 统一 Transformer 架构**。与以往依赖庞大预训练网络或固定帧数处理的方法相比，本架构通过模块共享与自适应池化实现参数高效，且原生支持任意长度的突发帧输入。
2. **设计增强型可变形对齐（EDA）模块**。相较于传统 PCD 或单次可变形卷积，EDA 采用多尺度分层结构从低分辨率向高分辨率传递偏移量，兼具隐式去噪与层次化对齐能力，显著提升复杂运动场景的鲁棒性。
3. **提出基于参考帧的特征富集（RBFE）机制**。与仅做单次 warp 或拼接的现有对齐方法相比，RBFE 在每级对齐后引入参考帧二次交互，结合 back-projection 与 SE 思想显式提炼高频互补残差以修正微对齐误差。
4. **设计无参考特征富集（NRFE）与循环突发采样（CBS）**。相比 BIPNet 的伪突发（pseudo-burst）生成，CBS 以 zigzag 邻域组织实现长程帧间通信，在不增加显著计算开销的前提下替代刚性融合机制。

## 方法详解
- **整体流水线**：RAW 突发输入 → EDA 多尺度去噪/对齐/参考富集 → 图像重建阶段（ABFP 固定 burst 维度 → NRFE 循环采样融合 → 渐进上采样）→ 高质量 sRGB 输出。全程 $L_1$ 损失端到端训练（低光任务额外加入 perceptual loss）。
- **增强型可变形对齐（EDA）**：采用 3 级编码器-解码器结构，从 level 3 最低分辨率开始对齐并逐级传递偏移量。每级先经 Burst Feature Attention（BFA，基于 Restormer 的 MDTA + GDFN）提取局部与非局部上下文，再送入 Feature Alignment（FA）模块。FA 基于调制可变形卷积计算偏移 $\Delta n$ 与调制标量 $\Delta a$，对齐公式为 $\bar{g}^b = W_{\text{def}}(g^b, \{\Delta n, \Delta a\})$，非均匀采样点 $(n_i + \Delta n_i)$ 通过双线性插值实现，有效缓解分数像素位移。
- **基于参考帧的特征富集（RBFE）**：将对齐后的帧特征 $\bar{g}^b$ 与参考帧特征 $\bar{g}^{b_r}$ 拼接后输入 BFF 单元。BFF 先通过 BFA 生成注意力特征 $g_a^b$，再经压缩 $g_s^b = W_s g_a^b$ 与扩展 $g_e^b = W_e W_s g_a^b$ 得到 squeezed 与 expanded 特征，最终融合输出：$g_f^b = g_s^b + W(g_a^b - g_e^b)$，隐式学习高频残差互补信息。
- **图像重建与 NRFE**：首先通过 Adaptive Burst Feature Pooling（ABFP）将任意 B 帧沿通道拼接后做 1D 平均池化，固定输出 8 帧特征。NRFE 利用 CBS 以 zigzag 方式组织邻域帧对，避免仅相邻帧交互的限制，再通过 BFF 融合并利用 Pixel-shuffle 进行渐进上采样，减少参数量与计算成本。
- **训练设置**：输入打包为 4 通道 RGGB 格式，各任务共享 EDA/NRFE 模块；学习率从 $1e^{-4}$ 余弦退火至 $1e^{-6}$；batch size 4；随机水平/垂直翻转增强；无额外预训练。

## 实验与结果
- **数据集**：SyntheticBurst（合成，46,839 序列，14 帧）、BurstSR（真实，5,405 训练块）、SID Sony subset（低光，6,500 块，burst 4-8 帧）、Grayscale/Color Burst Denoising（各自 20K 训练块，4 种噪声增益）。
- **基线**：DBSR, HighResNet, LKR, MFIR, BIPNet, AFCNet, LEED, KPN, MKPN, BPN 等。
-
