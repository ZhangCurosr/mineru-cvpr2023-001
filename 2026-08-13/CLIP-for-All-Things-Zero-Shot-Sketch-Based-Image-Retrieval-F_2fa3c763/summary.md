---
title: "CLIP-for-All-Things-Zero-Shot-Sketch-Based-Image-Retrieval-F"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Sain_CLIP_for_All_Things_Zero-Shot_Sketch-Based_Image_Retrieval_Fine-Grained_or_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:41:17"
field: "跨模态检索与细粒度识别"
keywords: ["zero-shot sketch-based image retrieval", "CLIP", "prompt learning", "fine-grained retrieval", "cross-modal matching", "ViT"]
innovations: ["首次将CLIP适配到ZS-SBIR，提出sketch/photo双提示+LayerNorm微调的高效适配范式", "提出相对距离分布KL散度正则化实现跨类别统一triplet margin的零样本细粒度训练", "提出patch shuffling triplet以轻量方式学习实例级结构对应"]
benchmarks: ["Sketchy", "TUBerlin", "QuickDraw Extended"]
---

# 论文速读：CLIP-for-All-Things-Zero-Shot-Sketch-Based-Image-Retrieval-F

## 一句话总结
本文首次将视觉-语言基础模型CLIP适配到零样本草图图像检索（ZS-SBIR）任务，通过提示学习（prompt learning）为草图和照片分别设计独立的视觉提示，并结合文本编码器分类损失；进一步针对更难的细粒度跨类别ZS-SBIR，提出相对距离分布对齐的正则化损失与patch shuffling技巧，在类别级和细粒度级均大幅超越既有SOTA。

## 研究问题与动机
- **数据稀缺阻碍通用SBIR训练**：草图社区长期受限于标注数据不足，促使研究转向零样本设置（ZS-SBIR），即训练类与测试类完全不相交（$\mathcal{C}^{\mathrm{S}} \cap \mathcal{C}^{\mathrm{U}} = \emptyset$）。
- **现有ZS-SBIR语义迁移手段薄弱**：先前方法主要依赖word2vec词嵌入或间接知识蒸馏进行跨类别语义迁移，缺乏丰富的多模态语义先验，上限受限。
- **细粒度跨类别ZS-SBIR仍是空白**：现有FG-SBIR多局限于单类别内实例匹配，跨类别（未见形状未知）的零样本细粒度匹配从未被系统探索，且面临（i）跨类别三元组相对距离不均匀、（ii）缺乏实例级结构对应约束两大挑战。
- **CLIP的开放词汇泛化与跨模态理解天然契合SBIR**：CLIP已蕴含丰富语义隐空间与海量图文对齐先验，若能适配到草图域，有望同时解决语义迁移与跨模态对齐问题。

## 核心贡献（创新点）
1. **首次将CLIP适配至ZS-SBIR领域**：用浅层视觉提示（sketch/photo各一组）注入CLIP ViT编码器首层，冻结主体权重仅微调提示与LayerNorm参数，保留CLIP泛化力；与直接微调整网或加线性探针的本质区别在于避免灾难性遗忘并维持开放词汇能力。
2. **用CLIP文本编码器替代word2vec构建分类损失**：用手工模板"a photo of a [category]"生成类别文本特征并施加交叉熵分类损失，与以往用离散词向量或直接FC头的方法相比，充分利用图文联合语义空间而非纯文本先验。
3. **相对距离分布KL散度正则化（$L_\delta$）**：针对细粒度跨类别训练中triplet margin $\mu$ 随类别变化显著的问题，最小化各类别相对距离分布$\mathcal{D}_c$两两之间的KL散度，使得单一全局margin即可适用所有类别；与之前需元学习 per-category margin的方法本质不同，本方法完全在零样本设定下无需额外支持样本。
4. **Patch shuffling结构化对应学习（$L_{PS}$）**：对草图与照片的$n \times n$ patch做相同/不同随机置换后构建三元组损失，迫使模型学习置换不变的实例级结构对应；与Sinkhorn最优传输等复杂方案相比，该设计更轻量且直接以triplet形式表达结构对齐。
5. **同时推进类别级与细粒度级两个Setting**：前者采用独立sketch/photo分支与提示，后者转为共享骨干+公共提示+hard-triplets，体现对不同粒度匹配需求有意识的架构分化。

## 方法详解
- **骨干与提示注入**：使用CLIP ViT-B/32，冻结全部权重$\theta$，仅 trainable 参数为（i）浅层视觉提示$\mathbf{v}^s, \mathbf{v}^p \in \mathbb{R}^{K \times d_p}$（每支K=3个提示向量），插入ViT首层patch序列；（ii）每层Layer Normalization的可训练参数$\{l_\theta^s, l_\theta^p\}$（动机来自Frankle等训练BN的有效性）。
- **Triplet ranking损失（类别级）**：$\mathcal{L}_{\mathrm{Tri}}=\max\{0, \mu + d(f_s, f_p) - d(f_s, f_n)\}$，$\mu=0.3$，$d(a,b)=(1-a\cdot b)$，正样本为同类别photo，负样本为异类别photo。
- **文本编码器分类损失**：对每类$c_j$用模板"a photo of a $c_j$"经T得到$f_t^j$，计算$\mathcal{P}(y|I)=\frac{\exp(\mathrm{sim}(f_I, f_t^y)/\tau)}{\sum_j \exp(\mathrm{sim}(f_I, f_t^j)/\tau)}$，总$\mathcal{L}_{\mathrm{cls}}^{\mathcal{I}} = \frac{1}{N}\sum_i -\log \mathcal{P}(y_i|I_i)$，$\mathcal{I}\in\{s,p\}$。
- **总体类别级损失**：$\mathcal{L}_{\mathrm{Tri}}^{\mathrm{ZS\text{-}SBIR}} = \mathcal{L}_{\mathrm{Tri}} + \lambda_1(\mathcal{L}_{\mathrm{cls}}^p + \mathcal{L}_{\mathrm{cls}}^s)$，$\lambda_1=0.5$。
- **细粒度扩展要点**：改用hard-triplets（负样本与anchor同属一类但不同实例）、共享ViT骨干与单一公共提示$\mathbf{v}$；加入两项额外正则：
  - **相对距离分布对齐**：$\mathcal{L}_\delta = \frac{1}{N_s(N_s-1)}\sum_{i,j}\mathrm{KL}(\mathcal{D}_i \| \mathcal{D}_j)$，其中$\mathcal{D}_c=\mathrm{softmax}\{\delta(s_i, p_i^+, p^-)\}_{i=1}^{N_s}$，$\lambda_3=0.1$。
  - **Patch shuffling triplet**：$\mathcal{L}_{\mathrm{PS}} = \max\{0, \mu_{ps} + d(f_{s^{\gamma_1}}, f_{p^{\gamma_1}}) - d(f_{s^{\gamma_1}}, f_{p^{\gamma_2}})\}$，$\lambda_4=1$；实验中$n=2$（$2\times 2$ patch）效果最佳。
- **总体细粒度损失**：$\mathcal{L}_{\mathrm{Tri}}^{\mathrm{FG\text{-}ZS\text{-}SBIR}} = \mathcal{L}_{\mathrm{Tri}}^{\mathrm{hard}} + \lambda_2(\mathcal{L}_{\mathrm{cls}}^s+\mathcal{L}_{\mathrm{cls}}^p) + \lambda_3 \mathcal{L}_\delta + \lambda_4 \mathcal{L}_{\mathrm{PS}}$，$\lambda_2=0.5$。

## 实验与结果
- **数据集**：Sketchy(ext)（104 train / 21 test）、TUBerlin（220 train / 30 test）、QuickDraw Extended（80 train / 30 test）；FG任务仅用Sketchy同一零样本划分。
- **评估指标**：ZS-SBIR用mAP@all / P@200（TUBerlin用P@100，Sketchy-ext用mAP@200）；FG-ZS-SBIR用Acc@1 / Acc@5。
- **最强结果（对比既有SOTA的提升幅度）**：
  - **类别级ZS-SBIR平均提升约24.8%**：Sketchy mAP=0.723 / P=0.725（此前SOTA ZS-PSKD[ViT]为0.645/0.502）；TUBerlin mAP=0.651 / P=0.732（此前SOTA ZS-Sketch3T未报此组合，ZS-TVT为0.662/0.671）；QuickDraw mAP=0.202 / P=0.388（此前SOTA ZS-TCN为0.140/0.298）。
  - **细粒度跨类别FG-ZS-SBIR提升约26.9%**：Sketchy Top-1=28.68% / Top-5=62.34%，大幅领先CC-DG（22.6/49.0）与CrossGrad（13.4/34.9）。
- **基线对比要点**：各种CLIP适配基线（B-FT灾难性遗忘；B-Lin线性探针优于SOTA但仍低于提示法；B-Deep/B-MM/B-Cond与简单B-IP差距不大）证明浅层独立sketch/photo提示+LayerNorm微调是性价比最高的适配策略。
- **泛化稳健性**：在训练样本比例（10%–100%）与训练类别数（20–104）变化下，性能下降远小于对比方法，验证CLIP零样本潜力。
- **消融**：去掉分类损失导致FG精度骤降至10.69%/16.32%；去掉$L_\delta$降3.75%；去掉$L_{PS}$降3.15%；$K=3$提示数最优；$2\times 2$ patch比$3\times 3$优1.2% Acc@1。
- **文本检索对比**：off-the-shelf CLIP关键词检索在Sketchy上mAP=0.523，显著低于本文ZS-SBIR（0.723）；在细粒度上关键词检索Acc@1仅18.68%，低于本文FG方法9.0个百分点，印证草图在细粒度建模上的优势。

## 相关工作脉络
- **ZS-SBIR先驱ZS-CVAE/ZS-CAAE（ECCV'18）**：基于sketch→photo翻译缩小域Gap；本文不再依赖翻译，直接利用CLIP语义先验。
- **ZS-GRL（CVPR'19）与ZS-CCGAN（CVPR'19）**：分别用gradient-reversal与对抗学习对齐域；二者依赖word2vec语义，本文以CLIP文本编码器替代，语义表征更丰富。
- **ZS-Sketch3T（CVPR'22）**：test-time训练适应测试分布；本文完全在训练时完成适配，无需TTT开销，且能推广至未见类别。
- **Multi-category FG-SBIR（ECCV'22）**：用meta-learning学习per-category margin；本文指出其依赖少量支持样本，不适合纯零样本设定，改用分布对齐正则实现单全局margin。
- **CC-DG（CVPR'19）与CrossGrad（ICLR'18）**：前者建universal sketch trait manifold，后者用硬三元组+类/域分类器；二者均为纯视觉-度量思路，本文引入语言先验和结构shuffling双重信号。
- **Prompt Learning for Vision（ICCV'22/IJCV'22）**：CLIP-Adapter/Tip-Adapter等做特征适配器或条件提示；本文强调shallow prompt + LayerNorm微调在保留泛化与适配效率间的平衡，并扩展到sketch-specific双提示设定。

## 局限性与未来方向
- **手工作文模板依赖**：分类头依赖"photo of a [category]"模板，对难以用名词短语描述的形状属性或跨文化草图风格泛化性未评估。
- **patch shuffling粒度敏感**：$n=2$最优，过大网格会使草图出现过多空白patch、导致嵌入混淆；对高分辨率或复杂结构对象可能失效。
- **细粒度仍受类别内实例数限制**：$\mathcal{L}_\delta$依赖每类足够多的sketch-photo对才能稳定估计相对距离分布，极少样本类别下正则估计方差较大。
- **未见对长尾/细粒度难例的定量分析**：论文侧重总体mAP/Acc，未深入讨论类别间差异与失败案例分析。
- **未来方向**：（i）将提示范式推广到草图检测、分割等下游；（ii）探索learned text prompt与手工模板的折中；（iii）结合大比例无标签photo的半监督设定；（iv）扩展到更多跨模态检索场景（video、医学草图等）。

## 研究启发与可借鉴点
- **"冻结主干+浅层视觉提示+微调LayerNorm"的三件套**可作为foundation model适配跨模态小样本任务的高效范式，值得迁移到其他缺少配对数据的模态对（如深度图-可见光、CAD-自然图像）。
- **用已知文本模板调用预训练多模态编码器的分类头**，避免了额外FC层带来的灾难性遗忘，是零样本/少样本条件下兼顾判别力与泛化的实用技巧。
- **KL散度对齐triplet相对距离分布**的思想可推广到其他需要统一margin的多类别度量学习场景（如fine-grained classification、cross-domain retrieval）。
- **Patch shuffling构造结构对应监督信号**不依赖昂贵的光流或关键点标注，对任意成对图像都可用，可复用到细粒度配准、跨模态匹配等任务。
- **实验设计上**：提供丰富的CLIP适配基线族（FT/linear probe/conditional/deep/multimodal/independent/deep prompt）并形成完整ablation，这种"基线矩阵+逐项消融"的设计值得借鉴。

## 关键术语表
- **ZS-SBIR（Zero-Shot Sketch-Based Image Retrieval）**：用未见类别的草图查询，从多类别照片库中检索同属照片的跨类别零样本检索任务。
- **FG-ZS-SBIR**：在ZS-SBIR基础上进一步要求实例级（同一类别内不同个体）匹配，难度更高。
- **Prompt Learning（提示学习）**：在预训练模型输入端注入可学习的连续向量，冻结主体权重以实现参数高效的下游适配。
- **Hard Triplet**：三元组中负样本与锚点属于同一类别但不同实例，用于细粒度-instance-level区分。
- **Relative Distance Distribution Alignment（$L_\delta$）**：通过对各类别triplet相对距离分布做KL散度最小化，使跨类别margin需求趋于一致。
- **Patch Shuffling（$L_{PS}$）**：对图像分块做相同/不同随机排列后构成相似-相异对三元组，强制模型学习结构位置不变性。
- **Open-vocab Generalization**：模型可直接处理训练时未见过的新类别文本描述并完成推理的能力。
- **Catastrophic Forgetting**：在预训练大模型上做全量fine-tuning时原有通用知识被破坏的现象。

## 可复现要素
- **数据集**：Sketchy（extended）、TUBerlin、QuickDraw Extended，均按文献既定划分使用，公开可下载。
- **代码/权重**：项目页 https://aneeshan95.github.io/SketchLVM/ 声明有代码与项目页（论文未直接给出GitHub链接，需从项目页获取）；CLIP ViT-B/32权重为OpenAI开源。
- **关键超参**：ViT-B/32，输入224×224，$\mu=0.3$，lr=1e-5，batch=64，epochs=60；提示维度$3\times 768$，插入首层；$\lambda_1=0.5, \lambda_2=0.5, \lambda_3=0.1, \lambda_4=1$；FG中patch数$n=2$。
- **实现环境**：PyTorch，单卡Nvidia RTX 2080-Ti（11GB）。
