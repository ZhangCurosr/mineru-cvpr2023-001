---
title: "Aligning-Step-by-Step-Instructional-Diagrams-to-Video-Demons"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Zhang_Aligning_Step-by-Step_Instructional_Diagrams_to_Video_Demonstrations_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:50:26"
field: "视频理解与多模态对齐"
keywords: ["多模态对齐", "对比学习", "视频-图像检索", "指令演示", "最优传输", "进度编码"]
innovations: ["提出三种专用对比损失解决示意图与在野视频的对齐问题", "引入SPRF正弦进度特征替代Transformer位置编码", "构建IAW大规模家具组装视频-示意图对齐数据集"]
benchmarks: ["IAW", "Video-to-Diagram Retrieval", "Diagram-to-Video Retrieval"]
---

# 论文速读：Aligning-Step-by-Step-Instructional-Diagrams-to-Video-Demons

## 一句话总结
本文提出一种基于对比学习的多模态对齐框架，将IKEA家具组装的逐步抽象示意图与在野YouTube视频片段进行跨模态对齐，并同步发布包含183小时视频与近8,300张示意图的IAW数据集，在视频→图检索与集合对齐任务上显著优于CLIP等基线。

## 研究问题与动机
- **海量在野视频噪声大**：DIY视频多由非专业人士拍摄，包含与装配无关的评论、视角切换或工具差异，难以直接用于机器人模仿或人工指导。
- **模态鸿沟特殊**：说明书示意图是无文字、黑白、高度抽象的图标，相邻步骤（如“方框叠方框”）视觉差异极细微，传统文本/音频-视频对齐方法无法直接迁移。
- **缺少任务级基准**：现有家具组装数据集（如IKEA ASM、IKEA-FA）侧重动作识别或3D解析，缺乏“视频片段↔步骤图”双向对齐的大规模标注数据。
- **多对一匹配复杂**：同一装配步骤在视频中常对应多个连续片段，标准infoNCE损失假设一对一样本分布，易产生训练偏差。

## 核心贡献（创新点）
1. **发布IAW数据集**：收集420款IKEA家具、1,005个在野视频（共183小时）及8,263步说明书示意图，通过AMT获得真值对齐标注，覆盖多变视角、光照与多人装配场景。
2. **设计三种任务专属对比损失**：Video-Diagram Contrastive Loss（JS散度适配多对一匹配）、Video-Manual Contrastive Loss（同手册候选池分类先验）、Intra-Manual Contrastive Loss（高斯序进约束拉开相邻步骤特征）。
3. **提出SPRF进度编码**：将视频相对播放进度与步骤序号映射为正弦/余弦向量注入特征空间，实测优于标准Transformer位置编码。
4. **支持集合级软对齐**：结合最优传输（OT）与动态时间规整（DTW）完成整段视频与整本手册的联合匹配，OT因容忍步骤执行顺序微调而略优。

## 方法详解
- **骨干网络**：图像端使用ImageNet预训练ResNet-50（解冻后三层），视频端使用Kinetics-400预训练Slowfast-8x8（整体冻结）；两端输出维度均投影至1024维并用L2归一化。
- **SPRF特征**：对时长$ t_{\mathrm{duration}} $的视频片段，进度$ r^V=(t_{\mathrm{start}}+t_{\mathrm{end}})/(2t_{\mathrm{duration}}) $；第$ j $步示意图进度$ r^I=j/M $。映射为$ (\sin(\pi r), \cos(\pi r)) $后拼接至模态特征，使进度相近的跨模态样本在余弦相似度上自然靠近。
- **Video-Diagram Contrastive Loss**：将batch内所有视频→图、图→视频的匹配概率向量$ p^{V2I}, p^{I2V} $与真实分布$ q $用Jensen-Shannon散度对齐，缓解large batch下多个视频片段竞争同一图示的many-to-one冲突。
- **Video-Manual Contrastive Loss**：采样视频片段后，将其候选池限制为同手册全部$ M_i $张步骤图，计算cross-entropy损失并按$ M_i/\sum M_b $加权，强化“只在本手册内匹配”的任务边界。
- **Intra-Manual Contrastive Loss**：对同手册$ M $张步骤图计算图→图softmax概率$ p^{I2I} $，用JS散度逼向其对应步骤索引的离散化高斯分布$ \mathcal{N}(j, \theta) $，使Embedding空间距离近似于步骤序号距离，提升相邻易混淆步骤的可分性。
- **集合对齐（Set Matching）**：构建$ N\times M $相似度矩阵$ s_{ij} $，经$ \alpha>1 $幂次放大与归一化得到代价矩阵$ C_{ij} $；通过Sinkhorn-Knopp求解熵正则化OT问题得到软匹配矩阵$ T^\star $；DTW则施加严格的前向时序约束，实验中OT效果更稳。

## 实验与结果
- **数据集划分**：训练/验证/测试集共30,876 / 6,871 / 11,103个视频片段，按视角、室内外、机位运动、装配人数均衡抽样，验证/测试视频完全未见。
- **评估任务与指标**：视频→图检索（Top-1 Acc、平均索引误差AIE）；图→视频检索（Recall@1/3、AUROC）；整集对齐（OT/DTW后评估）。
- **核心数字**：基于步骤图（Step）时，Ours+OT在Video-to-Diagram Top-1 Acc达**31.61%**（vs CLIP 19.61%，提升约9.68%），AIE降至**3.458**；Diagram-to-Video R@1达**26.62%**，AUROC达**0.626**。基于整页图（Page）时Top-1 Acc进一步升至**36.71%**，R@1达**18.28%**。
- **消融结论**：Video-Manual Loss贡献最大；SPRF移除后Top-1骤降约7个百分点；OT略优于DTW，说明实际装配常有步骤微序调整；使用页面图辅助训练（即使测试用步骤图）带来正则提升。

## 相关工作脉络
- **装配/指令数据集**（IKEA ASM、IKEA-FA、EPIC-Kitchens、YouCook2、LEGO）：多聚焦动作识别、文本叙事对齐或3D解析；本文首次面向“在野视频↔抽象步骤示意图”双向对齐任务。
- **多模态时序对齐**（Everingham字幕-人物、Han et al.文本-装配视频、Xu et al.草图-运动视频）：依赖语言或光流特征；本文处理无文字、图标化的说明书，且支持双向检索与集合软匹配。
- **对比学习跨模态**（CLIP、ActionCLIP、Clip-ViP、MoCo/SimSiam系列）：通用图文/文视频预训练；本文针对many-to-one匹配与步骤序进性定制损失，非通用表征学习。
- **Sketch-based Video Retrieval**：草图与示意图同为黑白图标，但Sports类运动视频具强光流特性；本文场景强调静态装配逻辑与渐进变化，方法更具通用性。
- **排序/对齐优化**（DTW、Optimal Transport）：传统强时序约束在装配视频常失效；本文验证软OT在弱序先验任务中的优势。

## 局限性与未来方向
- **强监督依赖**：需AMT人工标注真值对齐，扩展至其他家具品牌或通用DIY场景成本较高。
- **单模态局限**：仅利用视觉（视频+示意图），未融合说明书旁白、工具音效或文本步骤描述。
- **场景单一**：实验限定于IKEA家具装配，未验证复杂工业装配或动态多变对象场景。
- **未来方向**：弱监督/自监督对齐扩展；引入音频叙述或多模态融合；面向机器人模仿学习与人类装配实时引导系统的下游应用。

## 研究启发与可借鉴点
- **进度显式编码**：SPRF以正弦/余弦将相对进度映射为周期特征，简单有效且无需额外训练参数，可迁移至手术视频-步骤图、烹饪教程-流程图等有时序对齐需求的任务。
- **JS散度替代KL处理多对一**：在候选池存在重叠匹配时，JS散度比KL更稳定，适用于任意“一对多/多对一”跨模态检索训练。
- **同任务内软序进约束**：用高斯分布替代冲激函数对相邻步骤特征施加序进惩罚，可推广至文档章节匹配、分镜脚本对齐等连续语义任务。
- **OT优先于DTW的实践启示**：当实际执行允许步骤微调顺序时，放弃硬性时序约束、改用软OT匹配往往更鲁棒，值得在多模态对齐评测中对比验证。
- **页面图辅助训练策略**：用粒度更粗的页面图参与训练可提供正则化，测试时切回细粒度步骤图仍保持高性能，为多尺度标注稀缺场景提供实用技巧。

## 关键术语表
- **IAW (Ikea Assembly in the Wild)**：本文构建的大规模IKEA家具在野组装视频与说明书示意图对齐数据集，含183小时视频、8,263步示意图及人工真值标注。
- **SPRF (Sinusoidal Progress Rate Feature)**：将视频相对进度与步骤序号映射为正弦/余弦向量，作为时序先验拼接至模态特征以辅助对齐。
- **Video-Diagram Contrastive Loss**：基于JS散度的分布对齐损失，专门处理多个视频片段可能对应同一示意图的many-to-one匹配问题。
- **Video-Manual Contrastive Loss**：以同手册全部步骤图为候选池的交叉熵分类损失，强制模型仅在所属手册范围内进行匹配。
- **Intra-Manual Contrastive Loss**：约束同手册步骤特征分布逼近以步骤索引为均值的高斯分布，增大相邻易混淆步骤的Embedding距离。
- **Optimal Transport (OT)**：在视频片段与示意图集合间求解软匹配平面的优化方法，通过Sinkhorn迭代高效计算，容忍弱序先验下的非严格对齐。

## 可复现要素
- **数据集**：IAW，论文公开获取途径（项目主页），标注基于Amazon Mechanical Turk与Vidat工具。
- **代码/权重**：项目主页 https://davidzhang73.github.io/en/publication/zhang-cvpr-2023/ ；Vidat标注工具开源，模型代码未在论文中明确标注GitHub链接，需以项目页为准。
- **关键超参**：视频重采样30fps，10s segment切64帧（2.13s）clip；特征维度1024；温度τ=0.07；AdamW lr=5×10⁻⁴, wd=5×10⁻³；训练20 epochs，batch=128 clips；OT超参ε=4, α=7。
