---
title: "Connecting-Vision-and-Language-with-Video-Localized-Narrativ"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Voigtlaender_Connecting_Vision_and_Language_With_Video_Localized_Narratives_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:43:44"
field: "视频视觉-语言理解"
keywords: ["Video Localization", "Vision-Language", "Video Annotation", "Narrative Grounding", "Video Question Answering", "Multimodal Learning"]
innovations: ["提出分主体叙事的视频标注协议 VidLNs，解决多主体复杂交互的稠密词级 grounding", "构建 Video Narrative Grounding (VNG) 新任务与 OVIS-VNG/UVO-VNG 基准，聚焦同一名词多实例消歧", "基于高质量叙事自动生成 VideoQA 基准（Oops-QA），包含文本输出与位置输出两类问题"]
benchmarks: ["OVIS-VNG", "UVO-VNG", "Oops-QA"]
---

# 论文速读：Connecting-Vision-and-Language-with-Video-Localized-Narratives

## 一句话总结
本文提出了 Video Localized Narratives（VidLNs），一种新的视频多模态标注协议，通过让标注者在关键帧上为每个主体分别讲述故事并将每个词与鼠标轨迹对齐，实现了视频中复杂多主体交互的稠密视觉-语言 grounding。基于此标注构建了 Video Narrative Grounding（VNG）和 Video Question Answering（VideoQA）两个新评测基准，并提供了强基线模型。

## 研究问题与动机
- 现有的 Localized Narratives（ImLNs）在静态图像上取得了出色的词级视觉对齐效果，但直接扩展到视频会面临"时间赛跑"问题——标注者难以在视频播放过程中同时跟随多个主体并描述复杂交互。
- 现有视频视觉-语言数据集大多仅针对单一领域（如烹饪、第一人称视角），缺乏对多主体、多对象复杂交互场景的系统性标注，且大多数仅提供短语级或名词级的定位，而非每个词的稠密 grounding。
- 视频中的同一名词可能出现多次、指代不同对象（如两只不同的"鹦鹉"），需要利用上下文进行消歧，这对视频理解模型提出了新的挑战。
- 已有的 R-VOS 等任务使用简短的指代表达式描述单个对象，而真实场景中需要理解长自然语言描述中多个名词的指代关系并分别定位。

## 核心贡献（创新点）
- **提出 VidLNs 视频标注协议**：通过分主体叙事（per-actor narration）机制，让标注者针对每个主体在关键帧上分别讲述故事，避免了"时间赛跑"问题，能清晰描述涉及多个主体及被动对象的复杂交互事件。与 ImLNs 的本质区别在于从单帧静态扩展到视频时序场景，并引入分主体叙事策略实现复杂交互的解耦。
- **构建大规模通用域视频标注数据集**：标注了 OVIS、UVO 和 Oops 三个数据集共约 2 万视频、7.2 万个主体、170 万词，提供每个词（包括名词、动词、形容词）的鼠标轨迹 grounding。与 ActivityNet-Entities 等数据集相比，覆盖领域更广（通用而非特定领域）、词级 grounding 更稠密、叙事更长更丰富。
- **提出 Video Narrative Grounding（VNG）新任务与基准**：要求模型输入视频和带标记名词的自然语言描述，输出每个名词在视频帧上的分割掩码，核心挑战是处理同一名词多次出现时的上下文消歧。构建了 OVIS-VNG 和 UVO-VNG 两个基准，分别有 505/7587 个视频。
- **提出 VideoQA 新基准（Oops-QA）**：包含文本输出型问题（自由文本答案）和位置输出型问题（时空位置答案）两类，共 6.2 万个问题，答案需深度理解整个视频内容。文本输出问题基于 VQ²A 自动生成并经人工验证，位置输出问题基于鼠标轨迹自动构建。

## 方法详解
- **五步标注协议**：①观看视频理解故事；②选择并命名主体（actor）和被动对象；③为每个主体选取若干关键帧（均匀采样候选帧）；④为每个主体分别讲述故事：用完整自然语言句子描述该主体的属性、动作及与其他对象/主体的交互，同时在关键帧上用鼠标轨迹标注所提及的对象和动作位置；最后单独描述背景；⑤手动转录语音并使用时序对齐算法将转录文本与音频对齐，从而确定每个词对应的鼠标轨迹段。
- **对齐方法改进**：直接将手动转录文本与音频对齐（而非像 ImLNs 那样先 ASR 再对齐），显著提高了词-轨迹对齐精度。标注时被要求说话缓慢、在移动鼠标时停止说话，并可点击停止鼠标轨迹以避免误轨迹。
- **VNG 任务定义**：输入视频 + 带标记位置的名词的叙事文本，输出每个标记名词在每个视频帧上的分割掩码（segmentation mask）。评估指标为 J&F（mean IoU 与 boundary measure 的平均）。同一名词可多次出现指代不同对象，需通过上下文消歧。
- **ReferFormer-VNG 基线模型**：修改原始 ReferFormer（原用于 R-VOS），核心改动是将条件查询（conditional query）从整句特征改为目标名词各 token 特征的平均池化，使得同一名词的多次出现可通过分别提取对应 token 特征来生成分开查询，从而实现多实例消歧。训练流程：先在 COCO-PNG 上预训练，再在 UVO-VNG 训练集上主训练，可选地在 OVIS-VNG 训练集上微调。
- **VideoQA 基准构建**：文本输出问题通过 VQ²A 方法从 VidLN -caption 自动生成，保留答案 1-2 词的问题并去重，人工验证后保留约 27%（每视频约 6 个问题）；位置输出问题通过 spaCy 词性标注和句法分析，将主语相关的句子转换为 "where is..." 问题，以关联的鼠标轨迹段作为地面真值框基础，人工验证后保留约 52%（每视频约 2 个问题）。
- **位置输出评估方法**：对预测的 bounding box 同时要求满足召回准则（|b∩t|/|t| ≥ 0.5，t 为轨迹段）和精度准则（|b∩g|/|b| ≥ 0.5，g 为由轨迹段推导的近似对象包围盒）。

## 实验与结果
- **标注质量**：语义准确性极高——名词短语准确率 96.8%-97.7%，动词准确率 96.0%-97.9%；定位精度（与 ground-truth mask 比对）平均 73.2%，过滤不连通轨迹段后提升至 77.3%；相比 ImLNs（bounding box 上 54.8%-57.6%），VidLN 精度提升约 +25%（83.1%）。
- **VNG 基线结果**（J&F 分数）：
  - 简单基线：使用完整叙事输入在 OVIS-VNG 上 22.9、UVO-VNG 上 25.8；仅使用名词关键词在 OVIS-VNG 上 25.7、UVO-VNG 上 35.6
  - ReferFormer-VNG（最优）：OVIS-VNG 无微调 32.0、微调后 32.7；UVO-VNG 46.4
  - ablation 表明 COCO-PNG 预训练和 UVO-VNG 主训练均对性能有贡献
- **VideoQA 基线结果**：
  - 文本输出问题：PaLI-1.5B 零样本准确率 24.1%-25.1%，微调后 44.9%-49.0%（使用 1 帧 vs 3 帧）
  - 位置输出问题：ReferFormer-VNG 基线满足召回准则 66.7%、精度准则 53.9%、两者都满足 48.3%
  - 综合得分：50.8%（两项子任务均值）
- 数据集规模对比：UVO-VNG 有 7,587 视频和 43,058 个对象，约为最大 R-VOS 基准（Refer-YouTube-VOS）的 2 倍视频数和 3 倍对象数。

## 相关工作脉络
- **Localized Narratives（ImLNs, Pont-Tuset et al., ECCV 2020）**：本文方法的前身，用于静态图像的标注协议。ImLNs 将标注者语音与鼠标移动同步对齐实现词级 grounding，本文将其扩展到视频并引入分主体叙事策略解决时序约束问题。
- **Ego4D / Epic-Kitchens**：大规模第一人称日常/烹饪视频数据集，配有旁白。本文聚焦第三方视角的通用域视频，强调多主体复杂交互的解耦描述，且提供所有词的 grounding 而非仅部分名词。
- **Panoptic Narrative Grounding（PNG, Gonzalez et al., ICCV 2021）**：在图像上对描述性 caption 中的名词进行全景分割 grounding。VNG 任务类似但扩展到视频域且关注具体对象而非 stuff 类别。
- **Referring Video Object Segmentation（R-VOS）**：使用简短指代表达式定位单个对象。VNG 使用长自然语言描述且需处理同一名词多实例消歧，任务更贴近真实场景。
- **ActivityNet-Entities / YouCook2-BB**：视频描述中带名词短语的 bounding box grounding。前者仅标注 432 个高频名词，后者仅限烹饪领域且测试集才提供定位；本文提供每个词的鼠标轨迹 grounding 且覆盖通用域。
- **Video Question Answering（如 ActivityNet-QA, TVQA+, WildQA 等）**：现有 VideoQA 数据集多为自动生成的 QA 对或来自特定类型视频（教程、电视剧）。本文基于高质量手动标注的叙事自动生成 QA，且包含需要时空定位的位置输出问题。

## 局限性与未来方向
- 标注成本较高：每视频需人工标注约 75 个词及其轨迹，2 万视频的工作量巨大，限制了数据集进一步扩展的可能性。
- 标注主观性：不同标注者对"主要主体"的选择和叙事风格可能存在差异，尽管人工验证环节在一定程度上控制了质量。
- 依赖已有分割标注：VNG 基准构建需要 OVIS/UVO 的 ground-truth 分割掩码作为起点，对没有预存掩码的场景需额外手动标注。
- 当前基线模型性能仍有较大提升空间（VNG J&F 最高 46.4%，VideoQA 综合得分 50.8%），说明任务远未解决，需要更强的多模态理解模型。
- 关键帧选择策略（均匀采样候选帧）可能遗漏重要时刻，未来可探索自适应关键帧选择。

## 研究启发与可借鉴点
- **分主体叙事策略**可用于其他需要解耦多实体交互的视频理解任务（如视频因果推理、多实体追踪），其"先识别主体→再分别叙事"的思路值得迁移。
- **鼠标轨迹分段+自动语音-文本对齐**的方法论可推广到其他视频标注协议设计，特别是需要词级细粒度定位的场景。
- **VNG 任务中的多实例消歧机制**（ReferFormer-VNG 通过对不同名词 token 生成独立查询）对处理共指消解（coreference resolution）具有借鉴意义，可应用于其他需要实体级理解的视频任务。
- **基于高质量叙事自动生成 QA**的流程（VQ²A 生成→人工验证筛选）是一种高效构建视频理解基准的方法，可复用于其他多模态问答数据集的构建。
- **组合评估策略**（如位置输出同时评估召回和精度，并引入近似 ground-truth box 缓解轨迹不完整问题）展示了如何在标注噪声存在时设计合理的评估协议。

## 关键术语表
**Video Localized Narratives (VidLNs)**：本文提出的视频多模态标注协议，通过分主体叙事和关键帧鼠标轨迹实现视频中文本的词级视觉 grounding。
**Localized Narratives (ImLNs)**：原始的图像级标注协议，标注者边说话边移动鼠标，实现图像中每个词与区域的同步对齐。
**Video Narrative Grounding (VNG)**：新提出的视频理解任务，输入视频和带标记名词的叙事文本，输出每个名词在视频帧上的分割掩码。
**ReferFormer**：原始的引用视频目标分割模型，使用视觉编码器和文本编码器结合 decoder 预测目标掩码；本文修改为 ReferFormer-VNG 以适应 VNG 任务。
**J&F Measure**：视频目标分割评估指标，为 mean IoU（J）和 boundary F 分数的平均值。
**VQ²A**：Visual Question and Answer generation 方法，从图像/视频描述中自动生成问答对。
**spatiotemporal grounding**：时空定位，指在视频的时间维度和空间维度上同时定位目标对象。
**co-reference resolution**：共指消解，指在文本中识别指向同一实体的不同表达（如"the man"和"he"）的任务。

## 可复现要素
- **数据集**：OVIS-VidLN（607 视频）、UVO-VidLN（7,588 视频）、Oops-VidLN（12,128 视频），约 20,323 视频；源代码和数据集已公开于 https://google.github.io/video-localized-narratives/
- **代码/权重**：论文提供了 ReferFormer-VNG 基线模型的实现和训练细节（见 supplement），视频标注工具和基准数据集链接已公开
- **关键超参**：ResNet-50 backbone（ImageNet 初始化）；训练流程为 COCO-PNG 预训练 → UVO-VNG 主训练 → 可选 OVIS-VNG 微调；PaLI-1.5B（ViT-L16 + mT5-Large）用于 VideoQA 基线；输入 1-3 帧视频
- **标注细节**：每视频平均 3.54 个主体、75.1 词（23.0 名词、9.5 动词、8.5 形容词）；鼠标轨迹通过点击手动分段，语音-文本直接对齐
