---
title: "CloSET-Modeling-Clothed-Humans-on-Continuous-Surface-with-Ex"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Zhang_CloSET_Modeling_Clothed_Humans_on_Continuous_Surface_With_Explicit_Template_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:42:49"
field: "3D人体建模与动画"
keywords: ["clothed human modeling", "point-based representation", "template decomposition", "continuous surface features", "pose-dependent deformation"]
innovations: ["端到端显式服装模板分解实现姿态相关变形的准确建模", "在连续身体表面上学习点特征并通过重心插值避免接缝伪影"]
benchmarks: ["ReSynth", "CAPE", "THuman-CloSET"]
---

# 论文速读：CloSET-Modeling-Clothed-Humans-on-Continuous-Surface-with-Ex

## 一句话总结
CloSET 提出了一种基于点云的全身服装建模方法，通过在连续身体表面上学习特征并结合显式模板分解策略，实现了对姿态相关服装变形的准确建模，同时解决了现有方法中的接缝伪影问题。

## 研究问题与动机
1. **服装变形建模的挑战**：从静态扫描创建可动画化虚拟人需要建模不同姿态下的服装变形，但服装风格多样、人体姿态复杂，使这一任务极具挑战性。
2. **现有表征方法的局限**：网格（Mesh）受限于固定拓扑；隐式场过于灵活，难以满足身体结构先验，易产生伪影；点云方法虽灵活高效，但在UV平面上学习特征会导致身体部位间的间断伪影。
3. **模板表示的不足**：仅使用最小化服装的身体模板无法准确建模宽松服装的姿态相关变形；现有隐式模板学习方法需要两步流程，阻碍端到端训练。
4. **数据集匮乏**：现有数据集多为紧身服装或仿真数据，缺乏多样化真实服装的高质量扫描数据。

## 核心贡献（创新点）
1. **端到端显式模板分解**：将服装变形分解为显式的服装相关模板和姿态相关褶皱，使姿态相关变形能被更好地学习和泛化到未见姿态。与FITE的两步显式-隐式模板分解方法相比，本文实现端到端联合优化且模板包含更多细节。
2. **连续表面特征学习**：在身体表面上学习分层点特征并通过重心插值连续采样，建立了连续紧凑的特征空间，避免了POP等方法在UV平面上的接缝伪影，同时能捕捉细粒度细节和长距离部位相关性。
3. **高质量新数据集THuman-CloSET**：引入了包含2000+高质量扫描的真实世界服装数据集，涵盖15种多样化服装款式，促进该领域研究。

## 方法详解
**连续表面特征学习（Continuous Surface Features）**：
- 以T-pose的SMPL/SMPL-X模型作为身体模板，模板顶点数N=6890（SMPL）或10475（SMPL-X）
- **姿态编码器** $\mathcal{F}_p$：将模板顶点 $\mathbf{V}^t$ 作为输入点云，已拟合的姿态顶点 $\mathbf{V}^u$ 作为特征，通过PointNet++学习姿态相关几何特征 $\phi_p(\mathbf{v}_n^t) \in \mathbb{R}^{C_p}$
- **服装编码器** $\mathcal{F}_g$：在多服装设置下，通过服装码 $\phi_{gc}$ 学习服装相关特征 $\phi_g(\mathbf{v}_n^t) \in \mathbb{R}^{C_g}$，这些特征在所有姿态间共享
- **重心插值**：对表面上任意点 $p_i^t$，通过顶点的重心坐标 $b_i$ 连续采样特征：$\phi(p_i^t) = \sum_{j=1}^{3}(b_{ij} * \phi(v_{n_{ij}}^t))$

**基于点云的服装变形（Point-based Clothing Deformation）**：
- **显式模板分解**：将位移分解为两部分：
  - 服装模板位移：$\boldsymbol{r}_i^g = \mathcal{D}_g(\phi_g(p_i^t), p_i^t)$
  - 姿态相关褶皱位移：$\boldsymbol{r}_i^p = \mathcal{D}_p(\oplus(\phi_g(p_i^t), \phi_p(p_i^t)), p_i^t)$
  - 总位移：$\dot{\boldsymbol{r}}_i = \boldsymbol{r}_i^g + \boldsymbol{r}_i^p$
- **局部坐标变换**：在世界坐标系中计算点位置：$\boldsymbol{x}_i = \mathcal{T}_i \boldsymbol{r}_i + \boldsymbol{p}_i^u$，其中 $\mathcal{T}_i$ 是基于未服装身体模型的局部变换矩阵

**损失函数**：
- 数据项：Chamfer距离 $\mathcal{L}_p$ 和法向量L1距离 $\mathcal{L}_n$
- 正则化项：$\mathcal{L}_{rgl} = \frac{1}{M}\sum\|\boldsymbol{r}_i\|_2^2 + \frac{\lambda_{pd}}{M}\sum\|\boldsymbol{r}_i^p\|_2^2 + \frac{\lambda_{gc}}{N}\sum\|\phi_{gc}(\boldsymbol{v}_n^t)\|_2^2$
  - 第二项鼓励姿态相关位移尽可能小，使不变形保留在模板中
  - 第三项正则化服装码

## 实验与结果
**数据集**：
- **CAPE**：真实扫描数据，选择subject 03375的blazerlong和shortlong服装
- **ReSynth**：基于物理仿真合成数据，包含裙子、夹克等挑战性服装
- **THuman-CloSET**：新引入的真实扫描数据集，2000+扫描，15种服装，100个姿态用于训练，其余用于评估

**评估指标**：Chamfer Distance (CD, $\times 10^{-4} m^2$) 和 Normal Difference (NML, $\times 10^{-1}$)

**主要结果**：
- **ReSynth数据集**（Table 1）：CloSET使用1/8训练数据即达到优于使用全量数据的Baseline的性能，CD均值1.240（POP为1.356），CD最大值5.543（POP为7.339）
- **ReSynth outfit-specific**（Table 3）：在挑战性裙子/连衣裙服装上显著优于SkiRT，如felice-004的CD从6.45降至6.01
- **THuman-CloSET**（Table 2）：在真实宽松服装场景下，CloSET在3个代表性服装上均取得最优CD和NML成绩
- **消融实验**（Table 5）：在dress outfit上，完整方法CD=6.01，消融连续表面特征后CD=7.05，消融模板分解后CD=6.53，验证两模块均有效

## 相关工作脉络
1. **SCALE [36]**：早期点云服装建模方法，使用全局PointNet特征，缺乏细粒度信息；本文在此基础上改进为分层表面特征学习。
2. **POP [39]**：SOTA点云方法，使用单张细粒度UV图，但在UV边界产生接缝伪影；本文改用连续身体表面避免此问题。
3. **FITE [32]**：两阶段隐式-显式模板分解方法；本文实现端到端显式模板分解，避免两步流程限制。
4. **SCANimate [59] / SNARF [10]**：隐式身体建模方法，依赖可微分蒙皮；本文发现其在宽松服装和稀疏姿态下学习skinning weight存在病态问题。
5. **SkiRT [37]**：学习改进蒙皮权重的点云方法；本文与其在 challenging skirt/dress 上对比，展示模板分解优势。

## 局限性与未来方向
1. **模板skinning weight不准确**：导致裙子/连衣裙类服装的点分布不均匀，结合可学习蒙皮方案[37, 59]可缓解
2. **未利用时序信息**：当前方法未考虑相邻帧的姿态连续性， enforcing temporal consistency 是未来方向
3. **缺少物理约束**：可能出现自相交等伪影，引入physics-based loss[61]可改进
4. **宽松服装拟合困难**：THuman-CloSET中宽松服装的底模拟合存在挑战，需更鲁棒的拟合策略

## 研究启发与可借鉴点
1. **连续特征空间的构建技巧**：将点特征学习置于参数化身体表面上并通过重心插值实现连续采样，为其他3D点云任务提供可迁移的表征设计思路
2. **显式-隐式分解的端到端实现**：将复杂的服装变形解耦为共享模板和姿态相关变形，通过正则化项实现自动分解，这一思想可应用于其他变形建模任务
3. **轻量化数据策略**：本文证明使用1/8训练数据即可达到SOTA，其分层特征提取和模板分解策略值得在数据稀缺场景下借鉴
4. **真实数据集构建经验**：THuman-CloSET采用密集相机阵列采集+SMPL-X拟合的流程，为类似数据采集提供方法论参考

## 关键术语表
**SMPL/SMPL-X**：常用参数化人体模型，提供T-pose模板顶点和蒙皮权重，用于表示身体姿态和形状
**Chamfer Distance**：双向点集距离度量，用于评估生成点云与地面真实点云的匹配程度
**Barycentric Interpolation**：重心坐标插值，在三角形网格表面上实现特征的连续采样
**PointNet++**：分层点云特征学习网络，通过多级采样和聚合捕捉多尺度局部结构
**Skirt/Dress Outfit**：挑战性宽松服装类型，其较大变形和复杂拓扑对建模方法提出更高要求
**Outfit-specific Setting**：针对单一服装类型的建模设置，与multi-outfit相比能学习更精细的服装特定变形

## 可复现要素
- **数据集**：CAPE和ReSynth公开可用；THuman-CloSET新项目页面提供
- **代码/权重**：项目页面 https://www.liuyebin.com/closet 提供代码和数据集
- **网络架构**：修改版PointNet++，6层特征抽象+6层特征传播（L=6）
- **关键超参**：$\lambda_{pd}$, $\lambda_{gc}$ 控制正则化强度（具体数值见Supp.Mat）
- **点数设置**：预测点云50K用于评估，模板顶点6890(SMPL)/10475(SMPL-X)
