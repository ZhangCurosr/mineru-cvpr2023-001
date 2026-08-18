---
title: "CLIP-for-All-Things-Zero-Shot-Sketch-Based-Image-Retrieval-F"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Sain_CLIP_for_All_Things_Zero-Shot_Sketch-Based_Image_Retrieval_Fine-Grained_or_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:41:19"
field: "草图图像检索"
keywords: ["零样本检索", "草图图像检索", "CLIP", "提示学习", "细粒度匹配", "跨模态检索"]
innovations: ["首次将CLIP适配于零样本草图检索，提出prompt learning+LayerNorm微调范式", "提出f-Divergence正则化稳定多类别三元损失的全局margin参数", "设计Patch Shuffling技术学习跨模态结构对应关系"]
benchmarks: ["Sketchy", "TUBerlin", "QuickDraw"]
---

# 论文速读：CLIP-for-All-Things-Zero-Shot-Sketch-Based-Image-Retrieval-F

## 一句话总结
本文首次将CLIP基础模型适配于零样本草图图像检索（ZS-SBIR）任务，通过提示学习（prompt learning）设计，在类别级和细粒度级两个设置下均大幅超越现有最优方法，分别提升约24.8%和26.9%。

## 研究问题与动机
- **数据稀缺问题**：草图社区长期受限于高质量草图数据不足，迫使研究聚焦零样本设置（ZS-SBIR），即训练集与测试集类别不相交，要求模型具备跨类别语义迁移能力。
- **语义迁移能力薄弱**：现有ZS-SBIR方法的语义迁移手段 rudimentary，多直接/间接使用word2vec词嵌入，缺乏丰富的跨模态语义理解。
- **细粒度匹配困难**：跨类别细粒度ZS-SBIR（FG-ZS-SBIR）要求实例级匹配，面临两大挑战：(i) 不同类别间草图-照片的相对特征距离不均匀，导致三元损失全局margin参数次优；(ii) 未见类别的形状形态未知，难以建立结构性对应关系。
- **基础模型的潜力未被探索**：CLIP具备丰富的语义潜在空间和跨模态理解能力，理论上非常适合SBIR任务，但如何适配（避免灾难性遗忘、保留泛化能力）尚未研究。

## 核心贡献（创新点）
- **首次将CLIP适配于ZS-SBIR**：提出基于prompt learning的适配范式，保持CLIP权重冻结，仅训练视觉提示和LayerNorm参数，避免灾难性遗忘，在类别级ZS-SBIR上超越所有Prior Arts约24.8%。
- **引入CLIP文本编码器的分类损失**：使用手工模板（如"a photo of a [category]"）构建类别文本特征作为分类权重，替代传统FC层分类头，增强跨类别语义迁移能力。
- **提出相对距离正则化损失（f-divergence）**：通过最小化跨类别相对距离分布的KL散度，使三元损失的全局margin参数适用于所有类别，解决细粒度匹配中的margin不均衡问题。
- **设计Patch Shuffling数据增强**：对草图和照片的n×n patch进行随机排列，通过triplet损失强制相同排列顺序的草图-照片对在特征空间中更接近，学习结构对应关系，无需复杂的Sinkhorn操作。

## 方法详解
**整体框架**：基于CLIP的ViT-B/32图像编码器，保持主体权重冻结，采用浅层提示学习（shallow prompt）。

**类别级ZS-SBIR设计**：
- **双模态视觉提示**：为草图分支（$\mathcal{F}_s$）和照片分支（$\mathcal{F}_p$）分别学习独立提示向量$\mathbf{v}^s, \mathbf{v}^p \in \mathbb{R}^{K \times d_p}$，注入Transformer第一层。
- **LayerNorm微调**：除提示参数外，微调每个Layer Normalization层的参数$\{l_\theta^s, l_\theta^p\}$，借鉴BatchNorm微调的有效性。
- **文本编码器分类损失**：通过CLIP文本编码器将模板"a photo of a [category]"编码为类别特征$t_j$，计算分类概率并施加交叉熵损失：
$$\mathcal{L}_{\mathrm{cls}}^{\mathcal{I}} = \frac{1}{N}\sum_{i=1}^{N} -\log \mathcal{P}(y_i|\mathcal{I}_i)$$
- **总损失**：$\mathcal{L}_{\mathrm{Tri}}^{\mathrm{ZS-SBIR}} = \mathcal{L}_{\mathrm{Tri}} + \lambda_1(\mathcal{L}_{\mathrm{cls}}^p + \mathcal{L}_{\mathrm{cls}}^s)$

**细粒度ZS-SBIR扩展**：
- **共享提示与骨干**：草图和照片分支共享单一提示$\mathbf{v}$和骨干网络，更适合实例级匹配。
- **Hard Triplet**：负样本取自同类别不同实例$(s_i^j, p_i^j, p_{\neq i}^j)$。
- **相对距离正则化损失**：定义相对距离$\delta(s, p^+, p^-) = d(s, p^+) - d(s, p^-)$，对每类别计算分布$\mathcal{D}_c = \mathrm{softmax}\{\delta\}$，最小化 pairwise KL散度：
$$\mathcal{L}_\delta = \frac{1}{N_s(N_s-1)}\sum_{i=1}^{N_s}\sum_{j=1}^{N_s}\mathrm{KL}(\mathcal{D}_i, \mathcal{D}_j)$$
- **Patch Shuffle损失**：对草图$s$和照片$p$按随机置换$\gamma$重排n×n patches得到$s^\gamma, p^\gamma$，构造triplet：
$$\mathcal{L}_{\mathrm{PS}} = \max\{0, \mu_{ps} + d(f_{s^{\gamma_1}}, f_{p^{\gamma_1}}) - d(f_{s^{\gamma_1}}, f_{p^{\gamma_2}})\}$$
- **总损失**：$\mathcal{L}_{\mathrm{Tri}}^{\mathrm{FG-ZS-SBIR}} = \mathcal{L}_{\mathrm{Tri}}^{\mathrm{hard}} + \lambda_2(\mathcal{L}_{\mathrm{cls}}^s + \mathcal{L}_{\mathrm{cls}}^p) + \lambda_3\mathcal{L}_\delta + \lambda_4\mathcal{L}_{\mathrm{PS}}$

**超参数**：输入分辨率224×224，margin $\mu=0.3$，学习率1e-5，batch size 64，训练60 epoch；提示维度$3\times768$；FG-ZS-SBIR中$n=2$（2×2 patches）。

## 实验与结果
**数据集**：
- Sketchy (extended)：104训练类，21测试类
- TUBerlin：220训练类，30测试类
- QuickDraw (extended)：80训练类，30测试类
- Sketchy fine-grained：用于跨类别FG-ZS-SBIR评估

**评估指标**：ZS-SBIR使用mAP@all、P@200（Sketchy-ext用mAP@200，TUBerlin用P@100）；FG-ZS-SBIR使用Top-1和Top-5准确率。

**主要结果**：
- **ZS-SBIR**：在Sketchy上mAP@all=0.723，P@200=0.725；TUBerlin上mAP@all=0.651，P@100=0.732；QuickDraw上mAP@all=0.202，P@200=0.388，平均超越SOTA约24.8%。
- **FG-ZS-SBIR**：在Sketchy上Top-1=28.68%，Top-5=62.34%，超越Prior Arts（CC-DG: 22.6%/49.0%）约26.9%。
- **泛化能力**：在不同训练数据比例（10%-100%）和不同训练类别数（20-104类）下性能稳定。

**消融实验**：
- 移除LayerNorm微调：ZS-SBIR下降约2.5%，FG-ZS-SBIR下降约1.5%
- 移除分类损失：FG-ZS-SBIR严重下降至10.69%（丧失类别判别能力）
- 移除Patch Shuffle：FG-ZS-SBIR下降3.15%
- 移除f-Divergence：FG-ZS-SBIR下降3.75%
- 用word2vec替代CLIP文本编码器：ZS-SBIR下降4.57%，FG-ZS-SBIR下降0.172（Acc@1）
- 学习文本提示：性能下降2.36%/0.078，手工模板泛化更好
- 提示数量K=3最优，K=4饱和

## 相关工作脉络
- **ZS-SBIR基线（ZS-GRL, ZS-CAAE, ZS-CVAE等）**：采用word2vec语义嵌入或图像翻译缩小域间隙；本文用CLIP的开放词汇泛化能力替代，无需域适应机制。
- **FG-SBIR方法（SketchNet, Deep Spatial-Semantic Attention）**：针对单类别或固定类别训练；本文首次探索跨类别零样本细粒度检索。
- **通用零样本方法（CrossGrad, CC-DG）**：CC-DG建模通用草图特征流形但需supporting pairs；本文完全零样本，无需额外支持样本。
- **CLIP适配方法（ViLBERT, BLIP等）**：多为全参微调或添加MLP适配器；本文用prompt learning+LayerNorm微调实现参数高效适配。
- **提示学习方法（VPT, CoOp等）**：VPT在ViT中注入浅层提示；本文区分草图/照片双模态提示，并扩展至细粒度任务。
- **测试时训练方法（ZS-Sketch3T）**：利用测试集重建进行域适应；本文无测试时计算开销，更适合实际部署。

## 局限性与未来方向
- **提示容量上限**：实验显示K>3后性能饱和甚至下降，提示过长可能过拟合CLIP，限制更复杂任务的适配。
- **细粒度提升空间**：FG-ZS-SBIR相对类别级的提升幅度（26.9% vs 24.8%）略低，说明结构对应学习的挑战仍存。
- **未探索多模态提示**：本文比较了独立提示、多模态提示（MM）、条件提示（Cond），但未深入探索更复杂的提示交互机制。
- **数据规模限制**：仅在3个公开数据集验证，未在大规模真实场景测试CLIP的零样本泛化边界。
- **未来方向**：可扩展至其他草图相关任务（草图生成、草图检测等），探索更多基础模型（如LLaVA、Flamingo）与草图的协同。

## 研究启发与可借鉴点
- **Prompt Learning + LayerNorm微调的组合策略**：冻结大模型主体、仅调提示和LayerNorm参数的范式，兼顾训练稳定性和性能提升，可迁移至其他跨模态检索任务。
- **CLIP文本编码器作为分类头**：用手工模板+文本编码器替代FC分类层，保留零样本泛化能力，比learned text prompt更稳定，值得在其他视觉-语言任务中尝试。
- **f-Divergence正则化稳定多类别训练**：通过KL散度对齐跨类别分布，使单一超参数适用于多类别场景，可推广至其他多类别对比学习任务。
- **Patch Shuffling学习结构对应**：通过排列不变性训练结构感知表征，计算低成本且有效，可借鉴于细粒度匹配、跨模态对齐等任务。
- **实验设计的完备性**：提供大量消融（提示数量、文本/视觉提示、LayerNorm作用等）和泛化分析（数据量、类别数变化），为后续研究树立标杆。

## 关键术语表
- **ZS-SBIR（Zero-Shot Sketch-Based Image Retrieval）**：零样本草图图像检索，查询为草图、目标为照片，且测试类别未在训练中出现。
- **FG-ZS-SBIR（Fine-Grained ZS-SBIR）**：细粒度零样本草图检索，要求在同一类别内匹配具体实例而非仅类别级别。
- **Prompt Learning**：提示学习，在预训练模型中输入层注入可学习连续向量，适配下游任务同时保持模型泛化能力。
- **Shallow Prompt**：浅层提示，仅注入Transformer第一层的prompt，相比deep prompt（多层注入）更简单高效。
- **f-Divergence Regularization**：f-散度正则化，本文特指KL散度，用于对齐跨类别的相对距离分布，稳定多类别训练。
- **Patch Shuffling**：Patch洗牌，将图像划分为n×n网格并随机重排，构造增强triplet学习结构对应关系。
- **LayerNorm微调**：仅微调Layer Normalization层参数，保持主干网络权重冻结，平衡适配效果与灾难性遗忘。
- **Hard Triplet**：硬三元组，triplet中负样本与正样本同属一类但不同实例，用于细粒度实例级匹配训练。

## 可复现要素
- **数据集**：Sketchy (extended)、TUBerlin、QuickDraw (extended) 均公开可下载。
- **代码开源**：论文提供Project page（https://aneeshan95.github.io/SketchLVM/），但代码未明确声明开源仓库。
- **预训练权重**：使用CLIP ViT-B/32公开权重。
- **关键超参**：输入224×224，margin μ=0.3，lr=1e-5，batch=64，epochs=60，提示维度3×768，λ₁=0.5, λ₂=0.5, λ₃=0.1, λ₄=1，patch shuffle使用2×2网格。
