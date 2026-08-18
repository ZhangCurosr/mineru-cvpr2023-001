---
title: "BAD-NeRF-Bundle-Adjusted-Deblur-Neural-Radiance-Fields"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_BAD-NeRF_Bundle_Adjusted_Deblur_Neural_Radiance_Fields_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:51:59"
---

# 论文速读：BAD-NeRF-Bundle-Adjusted-Deblur-Neural-Radiance-Fields

## 一句话总结
BAD-NeRF提出了一种针对严重运动模糊图像的束调整去模糊神经辐射场，通过显式建模模糊图像的物理成像过程，在NeRF训练过程中联合优化三维场景隐式表示与曝光时间内的相机运动轨迹，从而在位姿不准确与图像严重模糊的条件下实现高质量去模糊与新视角合成。

## 研究问题与动机
1. **假设失效**：传统NeRF假设输入图像清晰且相机位姿精确，但低光等现实场景易产生运动模糊，直接破坏“锐利图像”假设。
2. **位姿估计瓶颈**：运动模糊导致特征点检测与匹配困难，使COLMAP等常规工具估计的相机位姿严重失准，进一步劣化NeRF重建质量。
3. **现有方法局限**：Deblur-NeRF等前期工作依赖外部预估计位姿（COLMAP或GT），且使用可学习的点扩散函数（PSF）卷积建模模糊，无法处理深度突变处的遮挡伪影，对位姿精度高度敏感。
4. **物理先验可利用**：单次曝光时间通常极短（≤200ms），期间相机运动可近似为SE(3)空间内的线性轨迹，为联合优化场景与连续轨迹提供了可靠的几何先验。

## 核心贡献（创新点）
1. 提出了基于NeRF框架的光度束调整 formulation，将运动模糊物理成像过程（时间积分平均）嵌入可微渲染管线；与Deblur-NeRF依赖固定位姿及可学习PSF卷积建模不同，本文通过沿轨迹渲染并平均虚拟锐利图像来模拟模糊，更贴合物理真实且能自然处理遮挡。
2. 设计了曝光时间内的连续相机运动轨迹建模方法，以SE(3)起始与终止位姿为可优化参数；与BARF等优化单时间点离散位姿的方法不同，本文显式建模了曝光窗口内的连续运动，大幅降低优化变量维度并提升轨迹一致性。
3. 实现了NeRF网络权重与相机轨迹的联合端到端优化，无需准确预估计位姿；与Park et al.基于显式多视图深度图优化的经典方法不同，本文利用隐式NeRF的强多视图一致性表征，在严重模糊下仍能保持几何完整并支持高质量新视角合成。

## 方法详解
- **神经辐射场表示**：沿用原始NeRF架构，使用两个MLP（$\mathbf{F}_\theta$）结合Fourier embedding将3D点坐标与世界坐标系视角方向映射至高维空间，输出颜色$c$与体密度$\sigma$，通过标准可微体积渲染计算锐利像素强度$I(\mathbf{x})$（对应公式4、5）。
- **模糊成像物理模型**：将捕获的模糊图像$\mathbf{B}(\mathbf{x})$建模为曝光时间$\tau$内一系列虚拟锐利图像$\mathbf{I}_t(\mathbf{x})$的积分平均：$\mathbf{B}(\mathbf{x}) = \phi \int_0^\tau \mathbf{I}_t(\mathbf{x}) dt$（公式6），离散近似为等权平均：$\mathbf{B}(\mathbf{x}) \approx \frac{1}{n} \sum_{i=0}^{n-1} \mathbf{I}_i(\mathbf{x})$（公式7）。该过程完全可微，使得梯度可从模糊损失反向传播至各虚拟锐利图像。
- **相机运动轨迹建模**：对每张模糊图像参数化曝光起始位姿$\mathbf{T}_{\mathrm{start}}$与终止位姿$\mathbf{T}_{\mathrm{end}}$（均属于SE(3)）。中间时刻$t$的虚拟相机位姿通过李代数指数映射线性插值得到：$\mathbf{T}_t = \mathbf{T}_{\mathrm{start}} \cdot \exp(\frac{t}{\tau}
