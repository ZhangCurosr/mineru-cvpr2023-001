---
title: "BlendFields-Few-Shot-Example-Driven-Facial-Modeling"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Kania_BlendFields_Few-Shot_Example-Driven_Facial_Modeling_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:41:02"
---

# 论文速读：BlendFields-Few-Shot-Example-Driven-Facial-Modeling

## 一句话总结
本文提出 BlendFields，一种仅需5张极端表情多视角图像即可驱动的少样本面部神经渲染方法。该方法通过在四面体网格上度量局部体积形变，动态混合多个表情专属的辐射场，从而在未见表情下生成皱纹等高频率细节，兼顾了数据效率、可解释控制与渲染保真度。

## 研究问题与动机
1. **高频细节缺失**：现有基于参数化模型（如VolTeMorph）的体积渲染受限于网格离散分辨率，无法真实还原表达式依赖的高频现象（如皱纹、肌肉挤压）。
2. **数据与算力门槛过高**：数据驱动方案（如AVA）需310万张图像与海量GPU算力，且数据集未公开，难以普惠研究社区。
3. **泛化与可控性矛盾**：NeRFies/HyperNeRF等隐变量方法需密集训练数据，且隐空间难以合理外推至未见序列或新表情。
4. **缺乏物理可解释的少样本驱动机制**：亟需一种结合传统CG先验、仅需少量极端样本即可生成逼真动态细节的建模范式。

## 核心贡献（创新点）
1. **将blend-shape corrective思想引入神经辐射场**：训练K个极端表情的残差辐射场并通过局部形变感知权重动态混合，本质区别在于以物理几何量替代纯数据驱动的隐变量插值。
2. **基于四面体体积变化的自适应混合机制**：利用形变前后雅可比行列式度量局部压缩/扩张，以此作为表情相似度的代理特征；与既有方法相比，无需额外标注，仅依赖可微网格形变即可实现空间自适应细节激活。
3. **顶点场Laplacian平滑去伪影策略**：针对离散混合权重导致的相邻四面体跳变，在四面体流形上施加单次扩散正则；区别于传统图像域模糊，该操作保持了几何拓扑约束与实时推理兼容性。
4. **跨域泛化验证**：除人脸外，成功将该框架迁移至橡胶圆柱弯曲/扭转场景，证明其对任意参数化网格驱动的可变形物体具有通用性。

## 方法详解
- **基础框架**：继承VolTeMorph的四面体网格驱动体积渲染管线，底层密度场 $\sigma$ 保持与中性表情一致，仅对辐射度 $\mathbf{c}$ 引入表达式条件化。
- **辐射场混合公式**：
  $\mathbf{c}(\mathbf{x}; \mathbf{e}) = \bar{\mathbf{c}}(\mathbf{x}) + \sum_{k=1}^{K} \alpha_k(\mathbf{x}; \mathbf{e}) \cdot \tilde{\mathbf{c}}_k(\mathbf{x})$
  其中 $\bar{\mathbf{c}}$ 为中性模板，$\tilde{\mathbf{c}}_k$ 为第 $k$ 个极端表情的残差场，$\pmb{\alpha}$ 为混合系数向量。
- **训练目标**：使用 $K$ 张极端表情多视角图像，训练时混合系数退化为 one-hot（$\alpha(\mathbf{x}) = \mathbb{1}_k$），最小化像素级MSE损失 $\mathcal{L}_{\mathrm{rgb}} = \mathbb{E}_k \mathbb{E}_\mathbf{r} \| C_K(\mathbf{r}; \mathbf{e}_k) - C_k(\mathbf{r}) \|_2^2$。
- **混合权重计算**：
  1. 局部体积变化描述子：$\mathcal{G}(\mathbf{v}(\mathbf{e})) = \bigoplus_{\mathbf{T} \in \mathcal{N}(\mathbf{v})} \Delta \mathcal{V}(\mathbf{T}(\mathbf{e}))$，其中 $\Delta \mathcal{V} = \det(\mathbf{D} \cdot \bar{\mathbf{D}}^{-1})$ 为四面体形变梯度行列式。
  2. 相似度度量：$\Delta \mathcal{G}_k(\mathbf{v}(\mathbf{e})) = \| \mathcal{G}(\mathbf{v}(\mathbf{e})) - \mathcal{G}(\mathbf{v}(\mathbf{e}_k)) \|_2^2$。
  3. 权重生成：$\pmb{\alpha}(\mathbf{v}(\mathbf{e})) = \mathrm{softmax}_{\tau}\{ \Delta \mathcal{G}_k \}$，温度 $\tau=10^6$ 近似硬选择，满足凸组合与稀疏激活性质。
- **平滑后处理**：$\mathbf{A}^{\mathrm{diff}} = (\mathbf{I} - \lambda_{\mathrm{diff}} \mathbf{L})^{-1} \mathbf{A}^{n}$，$\lambda_{\mathrm{diff}}=0.1$，单次backward Euler迭代，消除相邻区域权重冲突产生的闪烁伪影。
- **推理流程**：输入表达式参数 $\mathbf{e}$ → 驱动四面体网格 → 计算顶点 $\mathcal{G}$ 与 $\pmb{\alpha}$ → 线性插值查询残差场求和 → 单阶段光线积分（$N=192$）。几何计算开销可忽略，保持实时渲染潜力。

## 实验与结果
- **数据集**：公开 Multiface（4位受试者，≈38机位/人，≈190训练图/人）。选5个极端表情训练，留3个未见表情评估；另用Houdini/Blender生成橡胶圆柱弯曲（24帧）与扭转（72帧）合成数据。
- **基线**：NeRF、Conditioned NeRF、NeRFies、HyperNeRF-AP/DS、VolTeMorph₁、VolTeMorph_avg。
- **指标**：PSNR↑、SSIM↑、LPIPS↓（仅面部区域）。
- **定量结果**（Table 2）：
  - **真实数据-闲散表情**：BlendFields PSNR 27.60 / SSIM 0.906 / LPIPS 0.085，全面优于次优 VolTeMorph_avg（26.92 / 0.891 / 0.111）。
  - **真实数据-新姿态合成**：PSNR 29.74 / SSIM 0.931 / LPIPS 0.078，显著领先 VolTeMorph_avg（28.69 / 0.918 / 0.098）。
  - **合成数据**：PSNR 32.29 / SSIM 0.9882 / LPIPS 0.0231，较 VolTeMorph_avg（30.21 / 0.9815 / 0.0387）提升约 +2.08 dB PSNR，LPIPS 下降约 40%。
- **消融结论**（Table 3）：邻域大小 $|\mathcal{N}(\mathbf{v})|=20$ 在合成数据上最优；Laplacian平滑在全部数据集上稳定提升指标；去除平滑会导致视觉伪影（Fig. 4）。
- **定性结论**：BlendFields 是唯一能动态、空间自适应生成表达式相关皱纹的方法；VolTeMorph_avg 将皱纹视为视角相关噪声而平均掉，VolTeMorph₁ 仅重现训练表情纹理。

## 相关工作脉络
1. **NeRFies / HyperNeRF**：依赖每图学习隐潜码驱动形变，外推至未见序列困难且需密集数据；BlendFields 以参数化3DMM形变为显式先验，实现少样本泛化。
2. **VolTeMorph**：实时体积渲染标杆，但细节受限于网格分辨率；本文在其基础上叠加 expression-dependent corrective blendfields，补齐高频缺失而不增加网格面数。
3. **AVA**：全数据驱动的高保真数字人，需310万图像与专有算力；本文以公开数据集+5个极端帧实现接近的视觉质量，大幅降低数据门槛。
4. **D-NeRF / MoRF**：侧重无监督动态场估计或单视图形变重建；本文聚焦强
