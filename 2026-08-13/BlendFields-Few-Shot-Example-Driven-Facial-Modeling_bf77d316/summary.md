---
title: "BlendFields-Few-Shot-Example-Driven-Facial-Modeling"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Kania_BlendFields_Few-Shot_Example-Driven_Facial_Modeling_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:52:41"
---

# 论文速读：BlendFields-Few-Shot-Example-Driven-Facial-Modeling

## 一句话总结
提出 BlendFields，一种仅需 5 张极端表情多视图图像的少样本面部神经辐射场方法；通过测量四面体网格的局部体积变化来自适应混合各表情的残差辐射场，在保持基模型低频平滑形变的同时，成功还原出皱纹等高频、表达式依赖的细节，实现未见表情的可控高保真渲染。

## 研究问题与动机
1. **数据与算力瓶颈**：现有高保真人脸神经渲染方法（如 AVA）依赖数百万张同步标定图像与受控光源，数据不公开且训练成本极高，难以 democratize。
2. **低频基模型的细节缺失**：基于 3DMM/FLAME 参数驱动的 NeRF（如 VolTeMorph）虽支持实时可控动画，但其四面体网格仅能表达平滑形变，无法捕获皱纹、皮肤挤压等高频细节。
3. **少样本下的形变泛化难题**：直接为每帧学习隐式形变场（如 NeRFies/HyperNeRF）在极少训练样本下极易过拟合，且缺乏物理可解释的控制信号。

## 核心贡献（创新点）
1. **Few-Shot 高频细节重建**：将 VolTeMorph 扩展至少样本设定，仅需 $K=5$ 张极端表情图像即可在未见表情上合成逼真的皱纹与皮肤形变。
2. **基于局部体积变化的辐射场混合机制**：首次将传统 CG 中的 blendshape corrective 思想引入神经场，用四面体 FEM 体积变化率 $\Delta\mathcal{V}$ 作为几何相似度度量，动态生成稀疏混合权重 $\alpha$，替代黑盒形变场。
3. **数据高效且跨域泛化**：相比 AVA 的 310 万张图像，本方法仅需约 190 张/人；除人脸外，成功泛化至 Houdini 模拟的橡胶圆柱体弯曲/扭转皱纹重建，验证了方法的普适性。

## 方法详解
- **基础架构与辐射场分解**：以 VolTeMorph 的四面体笼为支撑，密度场沿用中性表情的静态场 $\bar{\sigma}$，通过映射 $\bar{\mathbf{x}} = \mathcal{T}(\mathbf{x}; \mathbf{e} \to \bar{\mathbf{e}})$ 将任意表达式 $\mathbf{e}$ 下的点变换至规范空间。输出颜色分解为：
  $$\mathbf{c}(\mathbf{x}; \mathbf{e}) = \bar{\mathbf{c}}(\mathbf{x}) + \sum_{k=1}^{K} \alpha_k(\mathbf{x}; \mathbf{e}) \cdot \tilde{\mathbf{c}}_k(\mathbf{x})$$
  其中 $\bar{\mathbf{c}}$ 为模板外观，$\tilde{\mathbf{c}}_k$ 为第 $k$ 个极端表情的残差外观。
- **训练目标**：每个极端表情 $k$ 下，混合权重设为独热向量 $\alpha(\mathbf{x}) = \mathbb{1}_k$，仅对该表情的残差场施加 $L_{rgb}$ 重建损失（Eq. 8）。
- **自适应混合权重计算**：定义顶点局部体积描述符 $\mathcal{G}(\mathbf{v}(\mathbf{e})) = \bigoplus_{\mathbf{T} \in \mathcal{N}(\mathbf{v})} \Delta\mathcal{V}(\mathbf{T}(\mathbf{e}))$，其中 $\Delta\mathcal{V} = \det(\mathbf{D}\cdot\bar{\mathbf{D}}^{-1})$ 为四面体变形梯度行列式。测试时，计算当前表达式与各训练表达式的几何差异 $\Delta\mathcal{G}_k$，经极低温度 softmax（$\tau=10^6$）得到顶点混合权重：
  $$\pmb{\alpha}(\mathbf{v}(\mathbf{e})) = \mathrm{softmax}_{\tau}\{\Delta\mathcal{G}_k(\mathbf{v}(\mathbf{e}))\}$$
  实现“局部几何相似则激活对应表情残差”的稀疏激活，满足凸组合与 $L_0$ 稀疏性。
- **拉普拉斯平滑去伪影**：离散四面体间的权重跳变会导致视觉伪影，因此在四面体流形上对混合场 $\mathbf{A}$ 执行一步反向欧拉扩散：$\mathbf{A}^{\mathrm{diff}} = (\mathbf{I} - \lambda_{\mathrm{diff}}\mathbf{L})^{-1}\mathbf{A}^{n}$（$\lambda_{\mathrm{diff}}=0.1$）。
- **训练/推理设置**：训练单帧采样避免 OOM（batch=1024 rays，粗采样 128 + 重要采样 64，共 5×10⁵ 步，Adam lr=$5\times10^{-4}$，每 5×10⁵ 步衰减 0.1）；推理时利用底层网格单阶段采样，邻域大小取 $|\mathcal{N}(\mathbf{v})|=20$。

## 实验与结果
- **数据集与评估**：公开 Multiface 数据集 4 位受试者，每位约 190 张多视图图像（≈38 相机）。8 种手动选取的极端表情中 5 种训练、3 种（左右动嘴、鼓腮）用于未见表情测试。指标：PSNR、SSIM、LPIPS，分辨率 334×512。
- **定量结果（真实数据-新颖姿势合成）**：BlendFields 全面领先，**PSNR 29.74 / SSIM 0.931 / LPIPS 0.078**；次优基线 VolTeMorph_avg 为 28.69 / 0.918 / 0.098，相对提升显著。在自然表情序列插值任务中同样最优（PSNR 27.60 / SSIM 0.906 / LPIPS 0.085）。
- **跨域泛化实验**：在 Houdini 模拟的橡胶圆柱弯曲/扭转数据集上，BlendFields 准确捕捉了变形过程中的高频皱纹，合成数据达 **PSNR 32.29 / SSIM 0.988 / LPIPS 0.023**；基线方法在插值帧严重失真或残留训练态伪细节。
- **消融结论**：邻域大小 $|\mathcal{N}(\mathbf{v})|=20$ 时性能最佳；Laplacian 平滑在所有数据集上稳定提升质量；低对比度皱纹区域收敛较慢需更多步数。

## 相关工作脉络
1. **NeRF / D-NeRF / NeRFies**：依赖隐式形变场或每图潜码，难泛化至未见形变且少样本下易过拟合。本文摒弃端到端形变学习，改用显式 3DMM 驱动+稀疏残差混合。
2. **HyperNeRF**：通过高维表示建模拓扑变化，但需密集
