---
title: "Connecting-Vision-and-Language-with-Video-Localized-Narrativ"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Voigtlaender_Connecting_Vision_and_Language_With_Video_Localized_Narratives_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:43:49"
field: "视频理解与多模态 grounding"
keywords: ["Video Localized Narratives", "Video Narrative Grounding", "Video Question Answering", "Vision-Language", "Multi-modal Annotation", "Video Object Segmentation"]
innovations: ["提出 VidLN 标注协议实现词级视频时空 grounding，解决多主体复杂交互的去纠缠描述", "构建 VNG 和 VideoQA 两个新基准，引入同名消歧挑战", "VidLN 标注协议的定位精度（83.1%）相比 ImLNs（~55%）提升约 25%"]
benchmarks: ["OVIS-VNG", "UVO-VNG", "Oops-QA"]
---

# 论文速读：Connecting-Vision-and-Language-with-Video-Localized-Narratives

## 一句话总结
本文提出了**Video Localized Narratives（VidLN）**，一种将视频与语言连接的新型多模态标注协议：标注者对每个主体（actor）在关键帧上讲述故事并移动鼠标，从而为每个词提供精确的时空定位；基于该数据，构建了**视频叙事定位（VNG）**和**视频问答（VideoQA）**两个新基准，并在其上新建基线模型。

## 研究问题与动机
1. **现有图像级 Localized Narratives（ImLNs）无法直接扩展到视频**：直接在视频播放过程中同步说话和移动鼠标会导致"与时间赛跑"，通常只能跟踪一个显著对象，丢失复杂交互细节。
2. **视频领域缺乏词级（word-level）密集 grounding 的数据集**：既有视频 grounding 数据集要么仅覆盖特定领域（如烹饪、第一人称），要么只标注名词短语而非每个词，且不支持多主体复杂交互的去纠缠描述。
3. **VNG 任务中同名词的消歧是开放挑战**：同一段描述中多次出现的同一名词（如两个"parrot"）需要结合上下文进行消歧，现有 R-VOS 等方法无法处理此复杂度。
4. **VideoQA 需要更复杂的视频理解能力**：现有 VideoQA 数据集多为自动生成且缺乏人工校验，或仅针对单一场景（如电视剧），难以评估对多主体复杂交互视频的深层理解。

## 核心贡献（创新点）
1. **提出 VidLN 标注协议**：通过分 actor 讲述故事 + 关键帧操作的方式，在安静环境下完成视频的多主体叙事标注，避免"时间赛跑"问题；与 ImLNs 的本质区别在于引入了 per-actor 分解和关键帧静止操作，能清晰描述多主体间的复杂交互。
2. **构建大规模通用域 VidLN 数据集（20k 视频、170 万词）**：覆盖 OVIS、UVO、Oops 三个数据集，每个词均有鼠标轨迹 grounding，相比 ActivityNet-Entities 等数据集覆盖更通用的领域且 grounding 粒度更细（每个词 vs 仅 432 个常见名词）。
3. **提出 Video Narrative Grounding（VNG）任务及两个基准（OVIS-VNG、UVO-VNG）**：要求对输入叙事中每个名词输出视频帧级分割掩码，并引入同名消歧挑战；相比 PNG（图像级）和 R-VOS（短短语定位单个对象），VNG 处理长自然语言描述中的多重同名消歧。
4. **提出 Oops-QA 视频问答基准**：包含文本输出问答和时空位置输出问答两种类型，共 62k 问题，从高密度叙事自动生成并通过人工校验保证质量；相比 TVQA+ 等单一场景数据集，问题覆盖更复杂的跨主体交互视频。

## 方法详解
### VidLN 标注协议（5 步）
1. **理解视频**：标注者观看视频（可多次），理解其故事。
2. **选择主体（Actor Selection）**：标注者识别并命名视频中的主动对象（如"man"、"ostrich"），被动对象不单独标注。
3. **关键帧选择（Key-frame Selection）**：系统提供时间均匀采样的候选关键帧，标注者为每个 actor 选择覆盖其主要动作的几个关键帧。
4. **为每个 actor 讲述故事**：对每个 actor 单独进行——展示其关键帧，标注者用完整自然语言句子描述该 actor 的事件（包括属性、动作、与其他 actor/被动对象的交互），说话时移动鼠标指向对应的空间位置；最后用单独一行简要描述背景。关键指令：慢速说话，鼠标在对象间移动时暂停；可通过点击停止轨迹避免冗余。
5. **转录与时间对齐**：标注者手动转录语音（保证高质量文本），然后使用自动对齐算法 [2] 将手动转录直接对齐到音频，获得每个词的时间戳，从而确定对应哪段鼠标轨迹。

> 与 ImLNs 的关键技术改进：ImLNs 先做自动语音转文本再手动对齐，VidLNs 直接用手动转录对齐音频，效果更好。

### 评估指标
- **定位精度**：测量鼠标轨迹线段落在 ground-truth 掩码内的比例。平均精度 **73.2%**（分割掩码度量），剔除多连通分量后可提升至 **77.3%**。
- **语义准确性**：随机抽取 70 个视频，检查动词和名词短语是否正确描述视频内容，名词短语准确率 **97.7%/96.8%/97.2%**，动词准确率 **96.0%/97.8%/97.9%**（OVIS/UVO/Oops）。
- **VNG 评测**：采用 J&F 分数（mean IoU J 与边界 measure F 的平均）。
- **VideoQA 评测**：文本输出用 exact match accuracy；位置输出用基于近似 ground-truth box 的 recall（IoA ≥ 0.5）和 precision（IoA ≥ 0.5）双标准。

### Baseline 模型
- **ReferFormer-VNG**：将 ReferFormer（R-VOS 模型）适配到 VNG——将每个名词的 token features 做平均池化生成 conditional query，对同一名词的每次出现分别运行一次以产生不同 mask；使用 ResNet-50 视觉编码器，先在 COCO-PNG 预训练，再在 UVO-VNG 训练，可选在 OVIS-VNG 微调。

## 实验与结果
### 数据集统计
| 数据集 | 视频数 | 叙事数 | 平均每叙事 actor 数 | 平均每叙事词数 |
|---|---|---|---|---|
| OVIS-VidLN | 607 | 610 | 2.95 | 47.01 |
| UVO-VidLN | 7,588 | 8,587 | 3.00 | 63.97 |
| Oops-VidLN | 12,128 | 12,894 | 3.45 | 83.78 |
| **总计** | **20,323** | **22,091** | **3.54** | **75.06** |

### VNG 基准
- **OVIS-VNG**：505 视频，2,407 个带 mask 的名词（高遮挡挑战）
- **UVO-VNG**：7,587 视频，43,058 个带 mask 的名词
- 对比 R-VOS 基准：UVO-VNG 视频数几乎是 Refer-YouTube-VOS 的 2 倍，总对象数是其 3 倍

### VNG 实验结果（J&F 分数）
| 文本预处理方式 | OVIS-VNG | UVO-VNG |
|---|---|---|
| Full narrative | 22.9 | 25.8 |
| Noun only | 25.7 | 35.6 |
| **ReferFormer-VNG（full training）** | **32.7** | **46.4** |

- ReferFormer-VNG 最强结果：**UVO-VNG 测试集 J&F = 46.4**，OVIS-VNG 测试集 J&F = 32.7（经 OVIS-VNG 微调后从 32.0 提升至 32.7）
- COCO-PNG 预训练 + UVO-VNG 主训练是最强配置；去掉任一环节均显著降分

### VideoQA 实验结果
- **Oops-QA 文本输出**：PaLI-1.5B 零样本准确率 24.1%，微调后 44.9%（单帧）/ 49.0%（3帧）
- **Oops-QA 位置输出**：ReferFormer-VNG baseline recall 66.7%，precision 53.9%，两者均满足 48.3%
- **综合得分**：50.8%

### 与 ImLNs 的定位精度对比
- VidLNs 在 bounding box 上的 precision 为 **83.1%**（ segmentation mask 上 73.2%）
- 相比 ImLNs 在 Open Images 上的 57.6% 和 COCO 上的 54.8%，提升约 **+25%**

## 相关工作脉络
1. **Localized Narratives（ImLNs, Pont-Tuset et al., ECCV 2020）**：图像级标注方法，标注者边说话边移动鼠标为每个词 grounding；VidLNs 的核心灵感来源，但扩展至视频需解决"时间赛跑"和"多主体去纠缠"两大新挑战。
2. **Panoptic Narrative Grounding（PNG, Gonzalez et al., ICCV 2021）**：图像级任务，将 caption 名词映射到 panoptic segmentation；VNG 是其视频版，但专注具体对象而非 stuff 类别，且需处理同名消歧。
3. **Referring Video Object Segmentation（R-VOS, e.g., Refer-Former Wu et al., CVPR 2022）**：输入短短语定位单个对象；VNG 输入长自然语言描述，同名多次出现需消歧，任务复杂度更高。
4. **ActivityNet-Entities（Zhou et al., CVPR 2019）**：视频描述 grounding 数据集中最接近的现有工作，但仅 grounding 432 个最常见名词的 bounding box，且标注协议不支持多主体去纠缠；VidLNs 在每个词上提供更精细的鼠标轨迹 grounding。
5. **Ego4D（Grauman et al., CVPR 2022）/ Epic-Kitchens（ Damen et al., ECCV 2018）**：大规模第一人称日常视频数据集；VidLNs 聚焦第三人称视角的复杂多主体交互视频，领域互补。
6. **TVQA+（Lei et al., ACL 2019）**：视频问答数据集，在单一电视剧上评估时空 grounding；VidLN 的 VideoQA 使用来自多样源的复杂交互视频，答案可指代任意对象。

## 局限性与未来方向
1. **标注成本较高**：每视频约需标注 3.5 个 actor × 多关键帧，虽比全视频实时标注更高效，但仍是昂贵的人工标注过程，限制了数据规模进一步扩大。
2. **VNG 和 VideoQA 基准尚未被充分解决**：最强结果的 J&F 仅 46.4%（UVO-VNG），VideoQA 准确率约 50%，说明任务仍有较大提升空间，但也暗示当前方法离实用尚有距离。
3. **只覆盖通用领域**：未涉及电影、烹饪、第一人称等特定领域，不同领域可能需要调整标注协议。
4. **位置输出 QA 依赖关键帧选择**：位置输出问题仅评估在验证选定的单个关键帧，可能无法全面反映模型对视频时序的理解能力。
5. **潜在未来方向**：可扩展到更多视频数据集和领域；探索端到端的多模态预训练模型直接利用 VidLN 数据进行训练；将 VidLN 用于视频理解、视频生成等下游任务。

## 研究启发与可借鉴点
1. **分主体叙事标注协议是可迁移的范式**：将复杂视频拆解为 per-actor 故事线再分别 grounding 的思路，可应用于其他需要多实体交互理解的任务（如视频关系抽取、动作因果推理）。
2. **手动转录 + 直接音频对齐优于自动 ASR + 后对齐**：VidLNs 通过跳过自动转写步骤、直接用人工转录对齐音频，显著提升了 word-level 时间定位精度，这一数据质量控制策略值得在其他语音-视觉对齐任务中借鉴。
3. **利用现有 segmentation 标注 + 人工补充缺失部分**：VNG 基准构建时，先匹配 OVIS/UVO 已有的 mask 标注，再人工补充缺失部分，兼顾了效率与完整性，是构建新基准的实用策略。
4. **从稠密标注自动生成 QA 对 + 人工校验而非从零标注**：Oops-QA 使用 VQ²A 自动生成大量候选 QA 对，再通过人工验证保留高质量样本（仅保留约 27%），大幅降低了标注成本，该方法论可用于其他 QA 数据集构建。
5. **同名消歧作为视频理解的核心挑战**：VNG 明确将同名多次出现消歧作为任务核心难点，这一设计可为视频理解模型的研究提供新的评测维度和研究方向。

## 关键术语表
**Video Localized Narratives（VidLN）**：新型视频多模态标注协议，标注者在关键帧上为每个 actor 讲述故事并同时移动鼠标，实现词级时空 grounding。

**Video Narrative Grounding（VNG）**：新提出的视频理解任务，输入视频和带有名词标记的叙事文本，输出每个名词在每帧的分割掩码，核心挑战是同名消歧。

**Localized Narratives（ImLN）**：原始的图像级标注方法，标注者边说话边移动鼠标为图像的每个词提供 grounding 轨迹。

**J&F Measure**：视频对象分割的评估指标，为 mean IoU（J）与边界 measure（F）的平均值。

**ReferFormer**：基于 Transformer 的 R-VOS 方法，将文本作为条件 query 来预测视频帧中的对象分割掩码。

**PaLI**：Google 提出的多模态编码器-解码器模型，同时处理图像和文本输入，用于 VideoQA 基线实验。

**COCO-PNG**：基于 COCO 图像数据集构建的 Panoptic Narrative Grounding 数据集，用于 ReferFormer-VNG 的预训练。

**VQ²A**：从图像/视频描述自动生成问答对的算法，本文用于从 VidLN caption 自动产生 Oops-QA 的候选 QA 对。

## 可复现要素
- **数据集**：OVIS、UVO、Oops 三个数据集的 VidLN 标注已开源，地址：https://google.github.io/video-localized-narratives/
- **代码**：论文未提及代码开源链接
- **权重**：ReferFormer-VNG 基线模型的权重/训练细节见 supplement，论文主文未提供下载链接
- **关键超参**：视觉编码器使用 ResNet-50（ImageNet 初始化）；PaLI 使用 1.5B 版本（ViT-L16 + mT5-Large）；训练流程为先 COCO-PNG 预训练，再 UVO-VNG 主训练，可选 OVIS-VNG 微调；VQ²A 生成后保留答案长度为 1-2 词的 QA 对
