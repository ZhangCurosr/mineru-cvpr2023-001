---
title: "CloSET-Modeling-Clothed-Humans-on-Continuous-Surface-with-Ex"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_CloSET_Modeling_Clothed_Humans_on_Continuous_Surface_With_Explicit_Template_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:42:49"
field: "可动画化着装人体建模"
keywords: ["clothed human modeling", "point cloud representation", "explicit template decomposition", "continuous surface features", "pose-dependent deformation", "garment animation"]
innovations: ["端到端显式模板分解，将服装形变解耦为共享模板与姿态依赖皱纹", "在连续身体表面上学习点特征并通过重心插值采样，消除UV接缝伪影", "发布THuman-CloSET真实扫描数据集（2000+帧，15套多样服装）"]
benchmarks: ["ReSynth", "CAPE", "THuman-CloSET"]
---

# 论文速读：CloSET-Modeling-Clothed-Humans-on-Continuous-Surface-with-Ex

## 一句话总结
CloSET 提出了一种端到端点云驱动的方法，通过将服装形变解耦为显式服装模板与姿态依赖皱纹两部分，并在连续身体表面上学习特征，从而更好地建模真实场景下复杂服装的姿态依赖形变。

## 研究问题与动机
- **现有表示方法的局限**：网格受限于固定拓扑；隐式场过于灵活，难以满足身体结构先验，在未见姿态下易产生伪影；基于 UV 平面的点云方法（如 POP）在 UV 边界处存在明显的接缝伪影。
- **仅依赖身体模板不足**：SMPL 等身体模板仅建模最小着衣状态，对于宽松服装难以捕捉实际姿态依赖形变；隐式方法尝试学习 3D 空间中的蒙皮权重，但形变结果偏粗糙。
- **两步建模方案阻碍端到端学习**：FITE 等方法先学习隐式粗模板再学显式细节，需要两步流程，不利于端到端优化。
- **高质量真实扫描数据稀缺**：已有数据集或服装偏紧身，或依赖物理仿真合成，缺乏具有多样真实穿着风格的高质量扫描数据集。

## 核心贡献（创新点）
- **端到端显式模板分解（Explicit Template Decomposition, ETD）**：将服装形变分解为可学习的、跨姿态共享的显式服装模板位移与姿态依赖皱纹位移，与 FITE 的两步隐式模板方案本质区别在于端到端可微、模板细节更丰富。
- **在连续身体表面上学习点特征（Continuous Surface Features, CSF）**：基于 SMPL/SMPL-X T 形模板表面，通过重心插值实现特征的连续采样，消除了 POP 等方法在 UV 平面边界处的接缝伪影，与隐式 3D 特征场相比更加紧凑。
- **引入 THuman-CloSET 真实扫描数据集**：包含 2000+ 高质量扫描、15 套多样真实服装风格，弥补了已有数据集在宽松服装与真实拍摄方面的不足。

## 方法详解
- **连续表面特征学习**：输入为 SMPL/SMPL-X T 形模板顶点 $V^t$（共 6890/10475 个）与对应姿态顶点 $V^u$。姿态编码器 $\mathcal{F}_p$（基于 PointNet++）以 $V^t$ 为几何输入、$V^u$ 为特征输入，生成姿态依赖特征 $\phi_p(v_n^t)$。多服装训练时，服装码 $\phi_{gc}$ 经轻量 PointNet++ 编码器 $\mathcal{F}_g$ 得到服装相关特征 $\phi_g(v_n^t)$，两者对齐后合并为表面特征 $\phi$。
- **连续特征插值**：对模板表面上任意点 $p_i^t$，通过其所在三角形的重心坐标 $b_i$，由公式 $\phi(p_i^t) = \sum_{j=1}^{3}(b_{ij} * \phi(v_{n_{ij}}^t))$ 连续采样，特征不受顶点离散位置约束。
- **显式模板分解**：服装解码器 $\mathcal{D}_g$ 以 $\phi_g(p_i^t)$ 和 $p_i^t$ 为输入，预测服装模板位移 $r_i^g$；姿态解码器 $\mathcal{D}_p$ 以拼接后的 $\phi_g$ 和 $\phi_p$ 及 $p_i^t$ 为输入，预测姿态依赖皱纹位移 $r_i^p$。最终变形 $r_i = r_i^g + r_i^p$。
- **局部坐标变换**：位移 $r_i$ 在基于未着衣身体模型的局部坐标系中预测，再经变换矩阵 $\mathcal{T}_i$ 映射到世界坐标系：$x_i = \mathcal{T}_i r_i + p_i^u$，法向量同步变换。
- **损失函数**：总损失 $\mathcal{L} = \mathcal{L}_{data} + \lambda_{rgl}\mathcal{L}_{rgl}$。其中 $\mathcal{L}_{data} = \lambda_p\mathcal{L}_p + \lambda_n\mathcal{L}_n$，$\mathcal{L}_p$ 为归一化 Chamfer Distance，$\mathcal{L}_n$ 为 L1 法向量差异；正则项 $\mathcal{L}_{rgl} = \frac{1}{M}\sum\|r_i\|^2 + \frac{\lambda_{pd}}{M}\sum\|r_i^p\|^2 + \frac{\lambda_{gc}}{N}\sum\|\phi_{gc}(v_n^t)\|^2$，鼓励姿态无关形变保留在模板中。

## 实验与结果
- **数据集**：CAPE（紧身服装为主）、ReSynth（物理仿真）、THuman-CloSET（真实扫描，2000+ 帧，15 套服装）。
- **评估指标**：Chamfer Distance（CD，单位 $\times 10^{-4} m^2$）与法向量差异（NML，单位 $\times 10^{-1}$）。
- **ReSynth 数据集（表1）**：CloSET（仅用 1/8 训练数据）CD Mean = 1.240，Max = 5.543，NML Mean = 1.019，全面优于 SCALE（CD Mean 1.491）和 POP（CD Mean 1.356）。
- **ReSynth outfit-specific（表3）**：在 skirt/dress 类服装上，CloSET 在 christine-027 上 CD 1.49 vs SkiRT 1.54，在 felice-004 上 CD 6.01 vs SkiRT 6.45，均取得最优结果。
- **THuman-CloSET（表2）**：三套代表性服装（sweater/longshirt/skirt）上 CloSET 均优于 SCANimate、SNARF 和 POP，尤其在 skirt-005 上 CD 1.49 vs POP 1.66。SNARF 因宽松服装导致蒙皮权重学习失败。
- **消融实验**：CSF 在 CAPE 上 blazerlong CD 从 POP 的 0.78 降至 0.71；ETD 在 ReSynth dress 上 CD 从 POP 的 7.34 降至 6.01；两者结合效果最佳（表5）。

## 相关工作脉络
- **SCALE [36]**：点云表示服装的早期工作，采用全局特征而非层次化局部特征，细粒度信息不足；CloSET 在 Scale 基础上引入层次化 PointNet++ 与连续表面特征。
- **POP [39]**：点云 SOTA，使用单张高分辨率 UV 映射学习局部特征，但在 UV 岛屿边界存在接缝伪影；CloSET 放弃 UV 方案，在 3D 身体表面上学习连续特征。
- **FITE [32]**：两步建模（先隐式粗模板再显式细细节），非端到端；CloSET 将模板分解融入端到端训练，模板更精细。
- **SCANimate [59] / SNARF [10]**：隐式场方法，在宽松服装场景下面临 ill-posed 的蒙皮权重优化问题，CloSET 的显式分解策略有效规避。
- **SkiRT [37]**：通过学习 blend skinning 权重改进身体模板，与 CloSET 正交，CloSET 聚焦于特征空间连续性与模板显式分解。

## 局限性与未来方向
- ** skirt/dress 类服装的点分布不均匀**：因模板中蒙皮权重不准确导致，可与可学习蒙皮方案（SkiRT、SCANimate）结合缓解。
- **未利用相邻姿态的时序信息**：当前方法逐帧独立处理，引入时序一致性约束是可探索方向。
- **缺乏物理约束**：存在自相交等伪影，未来可引入物理损失（如 SNUG）进一步改善。
- **宽松服装下身体模型拟合困难**：THuman-CloSET 中服装宽松导致 SMPL-X 拟合精度受限，影响整体效果上限。

## 研究启发与可借鉴点
- **连续表面特征插值替代 UV 映射**：通过重心插值在参数化模板表面上学习特征，既可避免 UV 接缝伪影，又比 3D 隐式场更紧凑，可迁移至其他参数化人体/服装任务。
- **显式-隐式形变解耦设计**：将姿态不变部分（模板）与姿态依赖部分（皱纹）分别学习，有助于提高泛化能力；该解耦思路可应用于其他形变建模场景。
- **单帧独立处理 + 未来时序扩展**：当前无时序约束的设计便于快速迭代，也为引入视频级一致性学习预留了接口。
- **真实扫描数据集建设**：THuman-CloSET 展示了真实数据在宽松服装泛化上的价值，启发团队关注高质量数据采集而非仅依赖仿真数据。

## 关键术语表
- **CloSET**：本文提出的 Clothing on a continuous Surface with Explicit Template decomposition 方法。
- **Explicit Template Decomposition (ETD)**：将服装形变显式分解为跨姿态共享的服装模板位移与姿态依赖皱纹位移的端到端解耦策略。
- **Continuous Surface Features (CSF)**：在 SMPL/SMPL-X 身体模板表面上学习层次化点特征，并通过重心插值实现连续采样的特征空间。
- **Chamfer Distance (CD)**：衡量预测点集与真实扫描点集之间双向最近邻距离的评估指标。
- **ReSynth**：基于物理仿真生成的含多种挑战服装（裙子、外套等）的标注数据集。
- **THuman-CloSET**：本文发布的新数据集，包含 2000+ 真实扫描、15 套多样真实服装。
- **SMPL-X**：包含手部、面部表情的扩展身体参数化模型，本文用于 THuman-CloSET 的身体拟合。
- **PointNet++**：层次化点云特征学习网络，本文用于姿态编码器和服装编码器。

## 可复现要素
- **数据集**：CAPE 和 ReSynth 为公开数据集；THuman-CloSET 新数据集在项目页面提供（https://www.liuyebin.com/closet）。
- **代码与权重**：论文声明代码和数据集均在项目页面开源。
- **网络架构**：PointNet++ 基础，6 层特征抽象 + 6 层特征传播（L=6），远点采样索引仅在首次前向传播时计算并保存；pose/garment 编码器参数量与 POP 相当。
- **关键超参**：$\lambda_{rgl}$、$\lambda_{pd}$、$\lambda_{gc}$ 论文正文未给出具体数值，详见 Supplementary Material。
