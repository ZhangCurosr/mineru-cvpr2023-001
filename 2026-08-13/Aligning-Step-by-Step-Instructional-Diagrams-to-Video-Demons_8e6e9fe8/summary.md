---
title: "Aligning-Step-by-Step-Instructional-Diagrams-to-Video-Demons"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Aligning_Step-by-Step_Instructional_Diagrams_to_Video_Demonstrations_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:50:36"
field: "多模态视频理解与跨模态检索"
keywords: ["跨模态对齐", "对比学习", "视频检索", "说明书图解", "最优传输", "SPRF"]
innovations: ["提出SPRF正弦进度率特征建模视频-手册弱时序单调关系", "三类任务特定的对比损失（Video-Diagram/Video-Manual/Intra-Manual）解决多对一匹配与步间细粒度歧义", "基于熵正则最优传输的集合对齐模块，并验证OT优于DTW"]
benchmarks: ["IAW (Ikea Assembly in the Wild)", "Video-to-diagram Top1 Retrieval", "Diagram-to-video Recall@1/@3 & AUROC"]
---

# 论文速读：Aligning Step-by-Step Instructional Diagrams to Video Demonstrations

## 一句话总结
本文提出了一种监督对比学习框架，将IKEA家具组装步骤的抽象说明图与真实世界中由爱好者拍摄的视频片段进行跨模态对齐；为此构建了新的IAW数据集，并设计了三类任务特定的对比损失与正弦进度率特征（SPRF），在视频→图解检索上较CLIP提升9.68%，在完整视频与手册对齐任务上提升12%。

## 研究问题与动机
1. **核心问题**：如何将"在野"Web视频（DIY家具组装实录）与其对应的官方组装说明书中的逐步抽象示意图进行自动对齐。
2. **现有方法不足（文本/音频路线）**：已有工作多依赖文本或音频模态对齐视频片段，但说明书图解是高度抽象、黑白无字、强图形式表征，两者之间存在更大的语义鸿沟。
3. **步骤间细粒度相似性**：相邻步骤图示差异极小（如"把矩形放在另一个矩形上"），导致传统对比学习难以区分。
4. **缺乏标准化视觉语言**：不同手册没有统一绘制规范（同一部件可用不同比例矩形或编号表示），且组装动作对机器而言仍不可直接解析。
5. **现实场景复杂**：DIY视频含大量噪声——旁白、个人偏好、不同灯光/视角/人数/工具使用，需鲁棒的对齐机制而非简单截取。

## 核心贡献（创新点）
1. **IAW数据集**：首个包含183小时"在野"组装视频与近8,300张逐步说明图的监督对齐数据集，覆盖420种家具、14类别，标注来自Amazon Mechanical Turk（15,649段动作对齐）。
2. **三类任务特定对比损失**：Video-Diagram Contrastive Loss（JS散度代替KL，缓解infoNCE在多对一匹配下的假设失效）、Video-Manual Contrastive Loss（同手册内分类式对比，引入步骤数加权）、Intra-Manual Contrastive Loss（引导同手册步骤嵌入按顺序高斯分布散布，增强步间判别性）。
3. **正弦进度率特征（SPRF）**：利用视频时间进度与手册步骤进度的单调弱相关，构造半圆映射$(\sin(\pi r),\cos(\pi r))$作为时间先验拼接入特征。
4. **基于最优传输（OT）的集合对齐模块**：将单段检索升级为完整视频片段集合与手册步骤集合之间的概率匹配，并证明OT略优于DTW（后者有序约束在跳步/错序场景过于严格）。

## 方法详解
**整体架构**：视频编码器（ResNet-50-based Slowfast-8x8，Kinetics 400预训练，**冻结全部层**）与图像编码器（ResNet-50，ImageNet预训练，仅解冻前三层）各自提取特征，与SPRF拼接后经L2归一化、全连接投影至同维$D^\ell=1024$空间，以余弦相似度$f_\mathrm{sim}$度量。

**SPRF（式3-4）**：$r^V=(t_\mathrm{start}+t_\mathrm{end})/(2\cdot t_\mathrm{duration})$；$r^I=j/M$。映射到半圆$(\sin(\pi r),\cos(\pi r))$使进度相近者相似度更高。

**三损失**（温度参数$\tau$，JS散度$D_\mathrm{JS}$）：
- **Video-Diagram CL（式8）**：$D_\mathrm{JS}(p^{V\to I}\|q^{V\to I}) + D_\mathrm{JS}(p^{I\to V}\|q^{I\to V})$，处理多对一匹配；
- **Video-Manual CL（式9）**：在minibatch内把某视频片段的所有同手册步骤图作为候选，交叉熵$CE(p_i^{V\to I},p_i^{gt})$，权重$M_i/\sum_b M_b$倾向难样本（步骤多者）；
- **Intra-Manual CL（式11）**：同手册步骤图间softmax后与正态分布$\mathcal{N}(j,\theta)$做$D_\mathrm{JS}$，鼓励嵌入空间距离对应手册步序距离，$\theta$可学习。

**集合对齐（式12-13）**：成本矩阵$C_{ij}=(s_{ij}^\alpha-\underline{s}^\alpha)/(\overline{s}^\alpha-\underline{s}^\alpha)$，$\alpha=7$放大差异。熵正则最优传输：$\min\sum T_{ij}C_{ij}-\epsilon H(T)$，$\epsilon=4$，由Sinkhorn-Knopp求解，得到视频-图解联合匹配概率分布。实验表明OT略胜DTW。

## 实验与结果
**数据集划分**：训练30,876段/验证6,871段/测试11,103段；所有属性（视角、室内/外、摄像机运动、组装人数）均匀平衡，且验证/测试视频完全不见于训练。

**评估指标**：
- 视频→图解：Top-1 Acc.、平均索引误差AIE=$\frac{1}{N}\sum|j^*-j^{gt}|$；
- 图解→视频：Recall@1、Recall@3、AUROC（步骤图/整页图两种输入，分别记S/P）。

**关键结果（Tab.1）**：
| 方法 | V→D Top1 Acc.(S) | V→D AIE(S) | D→V R@1(S) | D→V R@3(S) | D→V AUROC(S) |
|---|---|---|---|---|---|
| Random | 5.66 | 9.33 | 6.58 | 19.90 | 0.375 |
| COSSIM | 11.89 | 4.36 | 12.43 | 32.90 | 0.561 |
| CLIP(infoNCE) | 19.61 | 4.27 | 16.94 | 38.67 | 0.590 |
| **Ours+w/OT** | **28.62** | **3.73** | **22.30** | **45.00** | **0.617** |
| Ours+S(整页) | 34.55 | 2.93 | 16.48 | 32.20 | 0.390 |

- 相对CLIP：V→D检索**Top1提升约9.68个百分点**，对齐任务（OT版）V→D Top1进一步升至31.61、D→V R@1达26.62。
- Ablation（Tab.2）：Video-Manual CL（B）贡献最大；SPRF移除后Top1从28.62跌至21.73；三损失联合（D1）为最优。

## 相关工作脉络
1. **EPIC-Kitchens / YouCook2**：视频与烹饪叙述对齐；本文聚焦家具组装而非烹饪，且对比模态从文本变为抽象示意图。
2. **IKEA ASM / IKEA-FA**：已有家具组装数据集，但目标多为动作/姿态识别，非视频-图解跨模态对齐。
3. **LEGO [37] / Shao et al. [28]**：从手册抽取可执行计划/3D模型；无需对应在野视频。
4. **Text-Video Retrieval (CLIP4Clip、T2VLAD、Hit)**：文本-视频方向成熟，但无法直接迁移至"黑白抽象图 vs 实拍视频"的鸿沟。
5. **Sketch-Based Video Retrieval [9] / Xu et al. [41]**：笔 sketch 与视频对齐；后者motion vector sketch 针对体育等特定类型，泛化受限；本文方法更通用且支持双向检索。
6. **Han et al. [19]**：组装视频与文本手册对齐；本文用**图**替代文本，挑战从语言理解转为图形语义。
7. **Contrastive Learning for vision-language (CLIP [27])**：验证了跨模态对比可行性，但CLIP未考虑多对一匹配、步间细微差异、时间进度先验——本文三损失正是为此定制。

## 局限性与未来方向
1. **依赖手工/MTurk标注**：15,649段对齐标注成本高，限制了规模扩展。
2. **仅覆盖IKEA家具**：420个SKU、14类，场景狭窄；不同品牌/产品类型手册风格各异。
3. **视频编码器完全冻结**：可能无法充分适配"在野"视频的域偏移。
4. **无音频/语言模态**：仅用视觉+图谱，未利用视频旁白（论文展望之一）。
5. **OT/D以全局矩阵为代价**：长视频/多步骤时显存压力大；未见在线流式方案。
6. **弱监督/无监督方向空白**：论文承认当前需GT对齐，未探索自监督或仅页面级弱标签设定。

## 研究启发与可借鉴点
1. **SPRF时间先验**：当两序列存在单调弱相关（如进度/步序）时，半圆映射的进度特征可作为通用先验拼接进跨模态对比表征；可迁移至任何"时间线对齐"任务（字幕-视频、代码-日志等）。
2. **Video-Manual 类内分组采样**：minibatch内以"同手册"约束候选集，配合步骤数加权，有效放大难样本信号——类似策略可用于任意"分组检索"场景。
3. **Intra-Modal 顺序散度损失**：用可学习正态分布代替delta作为顺序先验，既保序又提供宽松正则，适合任意顺序编码场景（步骤图、页码、时间戳）。
4. **OT 优于 DTW 的经验**：当对齐中存在跳步/错序/重复动作时，移除严格时序约束的最优传输反而更好——这一发现对装配、工艺等人类操作类任务具有普适参考价值。
5. **页面级图示辅助训练**：即便评测目标为"单步图"，训练时用"整页图"作为补充输入能带来正则效果，提示多粒度并行学习值得探索。

## 关键术语表
- **IAW (Ikea Assembly in the Wild)**：本文构建的数据集，包含183小时在野组装视频、420种家具、8,263个步骤图与15,649段人工标注的对齐动作。
- **SPRF (Sinusoidal Progress Rate Feature)**：将视频时间进度与手册步骤进度映射到半圆$(\sin(\pi r),\cos(\pi r))$的一维时间先验向量。
- **Video-Diagram Contrastive Loss**：基于JS散度的跨模态对比损失，处理多对一匹配、避免infoNCE假设失效。
- **Video-Manual Contrastive Loss**：在同手册候选内做的交叉熵分类式对比，步骤数加权以强化难样本。
- **Intra-Manual Contrastive Loss**：约束同手册步骤图在嵌入空间按正态顺序分布，提升步间判别力。
- **Optimal Transport (OT)**：在视频片段集合与手册步骤集合之间求概率联合匹配的最优传输，熵正则化(Sinkhorn)求解。
- **AIE (Average Index Error)**：预测步索引与真值步索引之差的均值，衡量"偏离程度"的细粒度指标。
- **COSSIM / CLIP基线**：仅用余弦相似度损失/infoNCE损失作用于配对特征的对照方法，骨干网络与本文保持一致。

## 可复现要素
- **数据集**：IAW，论文未公开下载链接（项目网站为 https://davidzhang73.github.io/en/publication/zhang-cvpr-2023/ ），标注来源AMT+Vidat；是否开源论文未明确声明。
- **代码/权重**：论文未提供开源代码与预训练权重。
- **骨干网络**：图像ResNet-50 (ImageNet)；视频Slowfast-8x8 (Kinetics 400)；两编码器均移除分类头。
- **特征维度**：1024；SPRF拼接前各模态特征L2归一化。
- **超参**：batch=128视频片段；lr=$5\times10^{-4}$；weight decay=$5\times10^{-3}$；AdamW；20 epoch；$\tau$ 默认0.07初始化且各损失独立可学；$\sigma$ 初始化=1。
- **对齐超参**：$\alpha=7,\ \epsilon=4$（OT正则）；训练~20h/A100-80GB。
- **数据增强**：视频随机resize crop；图解随机resize crop+水平翻转+旋转。
- **输入尺寸**：视频短边224、64帧/clip（2.13s）；图解长边224白填充。
- **视频分段**：10s segment后取连续64帧子clip，测试时对5个clip特征取平均。
