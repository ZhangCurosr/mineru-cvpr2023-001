---
title: "Towards Scalable Neural Representation for Diverse Videos"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/He_Towards_Scalable_Neural_Representation_for_Diverse_Videos_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:47:40"
field: "视频表征与压缩"
keywords: ["隐式神经表示", "视频压缩", "视频表征", "运动估计", "时序建模", "视频理解"]
innovations: ["解耦clip特定视觉内容与共享运动信息进行统一编码", "引入全局时序MLP与任务导向流联合建模时空冗余", "以INR作为高效数据加载器提升下游动作识别精度"]
benchmarks: ["UCF101", "UVG", "DAVIS"]
---

# 论文速读：Towards Scalable Neural Representation for Diverse Videos

## 一句话总结
本文提出 **D-NeRV**，一种面向大规模多样化视频的隐式神经表示框架，通过将每个视频clip解耦为"clip特定视觉内容"与"共享运动信息"分别建模，并引入时序推理与任务导向流，显著超越 NeRV 及传统/学习型视频压缩方法，同时可作为高效数据加载器提升下游动作识别性能。

## 研究问题与动机
- 现有 INR-based 视频编码方法（如 NeRV）仅适用于编码少量短视频，无法扩展到大规模、内容多样的视频集合。
- 简单将视频分块并用独立模型编码的策略无法利用跨视频长期冗余，性能劣于用单一共享模型编码所有视频（图2显示共享模型始终更优）。
- NeRV 将视觉内容与运动信息耦合建模，导致记忆多样化视频时难度急剧上升。
- 需要一种能在单一模型中高效编码长视频和/或大量多样化视频的新框架。

## 核心贡献（创新点）
1. **提出 D-NeRV 统一框架**：用单个神经网络编码大规模多样化视频；与 NeRV 将视频独立编码或串联编码的本质区别在于引入"clip条件+解耦建模"，直接面向多样视频扩展。
2. **解耦 clip 特定视觉内容与运动信息**：通过关键帧提取视觉内容并由共享解码器生成运动；区别于已有工作将所有内容仅靠网络权重记忆。
3. **引入全局时序 MLP（GTMLP）**：显式建模跨帧全局时序依赖；与 Transformer 等重型时序建模相比更轻量高效。
4. **采用任务导向流作为中间输出**：通过光流预估减少像素级空间冗余；区别于直接预测像素值的 NeRV/E-NeRV。
5. **验证 D-NeRV 可作为高效数据加载器**：在动作识别任务中以相同压缩比下取得 3%-10% 更高精度，并具备更快的解码速度。

## 方法详解
**整体框架**：D-NeRV 由视觉内容编码器 $\mathsf{E}$ 与运动感知解码器组成，输入为每个 video clip 的起止关键帧 $(I_0, I_1)$ 及所有帧索引，一次输出整个 clip。

**Visual Content Encoder**：将每个 clip 拆分为连续片段，采样起止关键帧送入卷积编码器 $\mathsf{E}$，得到多阶段内容特征 $\{I_0^l, I_1^l\}_{l=1}^L$，特征具有 clip 特异性。

**Motion-aware Decoder**：
- **Multi-scale Flow Estimation**：沿时间轴插值得到中间帧特征 $I_t^1$，拼接位置编码后输入流估计模块 $\mathcal{G}$，同时预测前向流 $F_{t\to 0}$ 与后向流 $F_{t\to 1}$（公式2）；通过双线性 warp 得到 $\hat{I}_{t\to 0}^l, \hat{I}_{t\to 1}^l$，以距离感知置信度加权融合为 $\hat{I}_t^l$（公式3-4）。
- **Spatially-Adaptive Fusion (SAF)**：将 $\hat{I}_t^l$ 经两个全连接层学习像素级调制参数 $\gamma_t^l, \beta_t^l$，对解码器特征 $M_t^l$ 做仿射调制 $J_t^l = \gamma_t^l M_t^l + \beta_t^l$（公式5-6），再经 Conv + GELU + PixelShuffle 上采样。
- **Global Temporal MLP (GTMLP)**：对 $T$ 帧特征 $O^l \in \mathbb{R}^{C\times H\times W\times T}$，沿通道在每个时间位置施加全连接层 $W^l \in \mathbb{R}^{C\times T\times T}$，残差方式建模全局时序依赖：$M^{l+1} = O^l + \text{matmul}(O^l, W^l)$（公式8）。
- **Final Stage**：拼接最终解码特征 $M_t^L$ 与 warped 帧 $\hat{I}_t$，经两阶段卷积得到重构帧 $I_t'$。

**Training**：损失为 L1 + SSIM 组合 $\mathcal{L} = \frac{1}{T}\sum_t \|I_t' - I_t\|_1 + \alpha(1-\text{SSIM}(I_t', I_t))$；关键帧使用现有图像压缩算法编码；训练时以 mini-batch 方式喂入整个数据集的连续视频 clip。

## 实验与结果
- **数据集**：UCF101（101类动作，13320视频）、UVG（7视频共3900帧）、DAVIS（视频修复）。
- **重建对比（UVG，Table 1）**：D-NeRV 平均 PSNR 达 **35.52 dB**，较 SOTA INR 方法 E-NeRV（32.12 dB）提升 **3.4 dB**；共享模型 NeRV*（31.87 dB）仍明显落后。
- **视频压缩（UCF101，Table 2）**：D-NeRV-L 达 **30.06 dB / 0.951 MS-SSIM**，较 NeRV-L（27.57 dB）提升 **2.5 dB**，并超过 H.264-L（28.54 dB）；S/M/L 各级均稳定超越 NeRV 与 H.264。
- **视频压缩（UVG，Figure 4）**：各 BPP 下 D-NeRV 均超越 NeRV **>1.5 dB**，且优于学习压缩方法 DVC 与 DCVC。
- **消融（Table 3）**：+SAF 带来 UVG +1.7 dB、UCF101 +2.8 dB；+Flow 再增 UVG +0.67 dB、UCF101 +0.5 dB。
- **时序建模对比（Table 4）**：GTMLP（31.44 dB）优于 Depthwise Conv、Attention（31.34 dB，但训练更慢）。
- **多样性影响（Table 7）**：从10类增至100类，D-NeRV PSNR 仅降 0.38 dB，而 NeRV 降 1.29 dB，证明 D-NeRV 对多样性视频更稳健。
- **动作识别（Table 8）**：以 D-NeRV 作为数据加载器，"Train" 设置较 NeRV 提升 **3-4%**，"Test" 设置提升 **6-10%**，且解码速度（fp32 383 VPS）远超 DCVC。
- **视频修复（Table 10）**：D-NeRV 平均 PSNR 21.3 dB，较 NeRV*（19.88 dB）提升 **1.4 dB**。

## 相关工作脉络
1. **NeRV [1]**：首个图像级 INR 视频表示，输入帧索引直接输出 RGB 帧；本文在共享模型与解耦设计上根本性超越其独立编码范式。
2. **E-NeRV [2]**：将 NeRV 的空间/时序上下文解耦；本文进一步将"内容-运动"在 clip 级别解耦并引入任务导向流。
3. **DCVC [18] / DVC [16]**：学习型视频压缩，遵循传统压缩管线并用 CNN 替换部分组件；INR 方法在训练流程与解码速度上具有优势。
4. **NRFF [12] / IPF [13]**：预测帧间运动补偿与残差；本文改用任务导向流（task-oriented flow）作为中间输出。
5. **H.264 / HEVC**：传统视频编码标准；D-NeRV 在同等 BPP 下达到更高 PSNR 与下游任务精度。
6. **Nirvana [9] / HNerv [10]**：CVPR 2023 后续工作，提出自适应网络与 autoregressive patch-wise 建模；本文聚焦大规模多样视频的统一编码范式。

## 局限性与未来方向
- 实验主要在 UCF101（101类）和 UVG（7视频）上验证，未在更大规模工业数据集（如 Kinetics、WebVid）上检验扩展性。
- 关键帧采样步长为 stride 8，对高速运动或大帧间变化的视频重建质量可能受限。
- GTMLP 虽轻量但仍为 $O(T^2)$ 复杂度，超长视频场景下需进一步优化。
- 论文主要关注压缩与重建，对极端低比特率下的语义保持能力未深入讨论。
- 未来可探索更轻量的时序建模（如线性 Attention、状态空间模型）、与 Transformer 视频表征的结合、以及更多下游任务（如分割、检测）的验证。

## 研究启发与可借鉴点
- **"内容-运动解耦"范式**可迁移至 3D 场景/动捕数据压缩、多视频联合表征等任务，降低模型记忆负担。
- **任务导向流（task-oriented flow）** 作为中间输出比直接预测像素更具语义保真性，值得在视频插值、超分等任务中复用。
- **GTMLP 的全局时序建模** 以极轻量代价逼近 Attention 性能，为视频理解模型中的时序模块提供高效替代方案。
- **INR 作为数据加载器** 的思路新颖：将视频压缩为可微表示后可加速训练 I/O，可与团队的视频预训练/自监督学习方向结合。
- **多尺度流估计 + 距离感知融合** 策略可用于需要跨帧一致性约束的任务（如视频修复、目标跟踪）。

## 关键术语表
- **Implicit Neural Representations (INR)**：用神经网络近似信号函数，将坐标映射为信号值（如 RGB），以权重隐式编码数据。
- **NeRV**：首个图像级 INR 视频表示模型，输入帧索引直接输出对应帧的 RGB 值。
- **D-NeRV**：本文提出的可扩展隐式神经视频表示框架，通过解耦内容与运动编码大规模多样视频。
- **Spatially-Adaptive Fusion (SAF)**：利用关键帧内容特征学习像素级仿射调制参数，有效融合 clip 特定内容到解码器。
- **Global Temporal MLP (GTMLP)**：沿时间轴对每通道施加全连接层，以残差方式建模全局时序依赖。
- **Task-oriented Flow**：面向视频增强/插值任务学习的光流，作为中间输出以减少空间冗余而非直接预测像素。
- **BPP (Bits Per Pixel)**：视频压缩中常用的比特率度量，表示每像素编码位数。
- **UVG / UCF101**：UVG 为标准 4K 视频压缩数据集（7视频）；UCF101 为包含 101 类人体动作的大规模动作识别数据集。

## 可复现要素
- **数据集**：UCF101、UVG、DAVIS（均已公开）。
- **代码/权重**：论文主页 https://boheumd.github.io/D-NeRV/ 注明有项目页，代码与模型权重开源（论文未明确 GitHub 链接，需参考项目页）。
- **关键超参**：AdamW 优化器，Cosine annealing LR schedule，batch size=32，lr=5e-4；UCF101 训练 800 epoch、warmup 160 epoch；UVG 训练 400 epoch、warmup 80 epoch；关键帧采样 stride=8；使用 Learnable Image Compression [53] 压缩关键帧。
- **模型尺寸**：UCF101 上 S/M/L 总大小分别为 79.2 / 94.5 / 114.5 MB（含关键帧压缩体积）。
