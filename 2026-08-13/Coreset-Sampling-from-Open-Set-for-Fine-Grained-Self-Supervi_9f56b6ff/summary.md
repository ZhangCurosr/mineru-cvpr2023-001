---
title: "Coreset-Sampling-from-Open-Set-for-Fine-Grained-Self-Supervi"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_Coreset_Sampling_From_Open-Set_for_Fine-Grained_Self-Supervised_Learning_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:44:32"
field: "自监督学习与细粒度视觉识别"
keywords: ["自监督学习", "细粒度识别", "Coreset采样", "开放集学习", "表征学习", "数据选择"]
innovations: ["提出OpenSSL问题设定，将细粒度目标数据集与无标注开放集结合进行自监督预训练", "设计SimCore算法，通过隐空间最近邻匹配从开放集中采样语义相关的coreset", "引入自适应停止准则，动态控制采样预算以避免无关样本污染"]
benchmarks: ["Aircraft", "Cars", "Pet", "Birds", "Dogs", "Flowers", "Action", "Indoor", "Textures", "Faces", "Food", "ImageNet-1k", "MS COCO", "iNaturalist", "Places365"]
---

# 论文速读：Coreset-Sampling-from-Open-Set-for-Fine-Grained-Self-Supervi

## 一句话总结
本文提出了**开放集自监督学习（OpenSSL）**问题设定，即在细粒度目标数据集预训练时，可利用大规模无标注开放集辅助学习。针对开放集与目标数据集之间的分布不匹配问题，作者设计了**SimCore**算法，从开放集中采样与目标集语义相似的子集（coreset），显著提升了自监督预训练效果及下游任务表现。

## 研究问题与动机
1. **细粒度任务依赖专家标注**：细粒度视觉识别（如飞机型号、鸟类分类）需要大量专业标注，成本高昂，现实中常缺乏或仅有少量标注数据。
2. **单一预训练策略不足**：仅在目标数据集上预训练受限于数据量；直接使用大规模开放集（如ImageNet）预训练可能因分布差异而降低目标任务性能（见图2）。
3. **开放集语义噪声问题**：开放集包含大量与目标域无关的样本，直接融合（X+OS）甚至可能损害表示学习质量。
4. **需要高效的子集采样策略**：通过人工 oracle 选择相关类别可带来显著提升，但实际场景无法获取标注，亟需无监督的采样算法。

## 核心贡献（创新点）
1. **提出OpenSSL问题设定**：首次明确将"细粒度目标数据集 + 大规模无标注开放集"结合用于自监督预训练，与现实场景中多源数据可用性契合。
2. **设计SimCore采样算法**：基于目标集在隐空间中的分布，通过迭代最近邻选择与目标集语义最接近的开放集样本，实现高效coreset构建。
3. **引入自适应停止准则**：通过比较每次采样子集的目标相似度与首轮的比值，动态决定采样预算，避免无关样本污染。
4. **系统性实验验证**：在11个细粒度数据集、7种开放集上，验证了SimCore在多种架构（ResNet、EfficientNet、ViT）、多种SSL方法（SimCLR、BYOL、DINO、MAE）下的泛化性与有效性。
5. **下游任务扩展验证**：证明了SimCore预训练的表征在kNN分类、半监督学习、目标检测、像素级分割、多属性分类等任务上均有效。

## 方法详解
**OpenSSL问题形式化**：给定细粒度目标数据集 $X$（有标注）和无标注开放集 $\mathcal{U}$，目标是利用两者进行自监督预训练，而非仅使用 $X$ 或简单拼接 $X \cup \mathcal{U}$。

**SimCore核心设计**：
1. **目标函数**：寻找子集 $S \subseteq \mathcal{U}$ 最大化 $f(S) = \sum_{x \in X} \max_{u \in S} w(x, u)$，其中 $w(x,u) = z_x^\top z_u$ 为余弦相似度，$z$ 为编码器 $E_\theta$ 输出的归一化特征。
2. **复杂度优化**：使用 k-means 聚类目标集 $X$ 得到质心集 $\hat{X}$（实践中 $k=100$），将目标替换为质心以降低计算开销（$\mathcal{O}(|\hat{X}| \cdot |\mathcal{U}|)$）。
3. **迭代采样策略**（Algorithm 1）：
   - 第 $t$ 轮从候选集 $\mathcal{U}_t$ 中选择距离 $\hat{X}$ 最近的样本作为 $S_t^*$；
   - 累积到coreset $\mathcal{T} = \bigcup_t S_t^*$；
   - 更新候选集 $\mathcal{U}_{t+1} = \mathcal{U}_t \setminus S_t^*$。
4. **停止准则**：计算 $\hat{f}(S_t^*) / \hat{f}(S_1^*)$，若比值小于阈值 $\tau = 0.95$ 则停止采样，防止引入与目标集差异过大的噪声样本。
5. **最终预训练**：将采样的coreset $\mathcal{T}$ 与目标集 $X$ 合并，重新初始化编码器进行SSL预训练。

## 实验与结果
**数据集**：
- 11个细粒度目标数据集：Aircraft、Cars、Pet、Birds、Dogs、Flowers、Action、Indoor、Textures、Faces、Food。
- 7个开放集：ImageNet-1k、MS COCO、iNaturalist 2021-mini、Places365、WebVision、WebFG-496及合并数据集ALL。

**评估基线**：
- 仅在目标集 $X$ 上预训练（baseline）
- 在开放集 $OS$ 上预训练
- $X + OS$（全量融合）
- $X + OS_{rand}$（随机采样1%/5%）
- $X + OS_{SimCore}$（本文方法）

**主要结果**（Table 2，ImageNet-1k作为开放集）：
- **平均提升**：SimCore + 停止准则相比仅用目标集预训练提升 **+10.5%**（11数据集平均），远超全量OS（+2.7%）和1%随机采样（+1.3%）。
- **最优单数据集**：Birds数据集从29.27%提升至37.65%（+8.38%）；Pet从59.23%提升至79.66%（+20.43%）。
- **停止准则效果**：各数据集实际采样比例从0.27%（Faces）到15.6%（Action）不等，体现了自适应优势。
- **架构/方法泛化**：在EfficientNet、ResNet18/101、ResNeXt50等架构（Table 3）及BYOL、SwAV、DINO、MAE等SSL方法（Table 4）上均一致提升。
- **不同开放集**（Table 5）：即使使用WebVision、WebFG-496等嘈杂开放集，SimCore仍显著优于无开放集预训练。

**下游任务**（Table 7）：
- **kNN分类**：20NN和200NN均获最优。
- **半监督学习**：10%/20%/50%标注比例下均领先。
- **目标检测**（mAP）：Aircraft提升18.8点（10.8→29.6）。
- **语义分割**（IoU）：Pet前景IoU提升0.9点，Birds前景IoU提升3.1点。
- **多属性分类**：全部任务最优。

## 相关工作脉络
1. **自监督学习分布不匹配问题**：El-Nouby等[21]指出ImageNet预训练不一定适用于不同域任务；Tian等[64]发现未经筛选的uncurated数据会损害SSL性能。本文区别于它们的方法是显式地从开放集**采样**而非降噪或蒸馏。
2. **开放集识别（Open-Set Recognition）**：Bendale & Boult [5, 12] 研究测试时拒绝未知类别。本文与之不同：开放集在**预训练阶段**使用，目标是利用相似样本增强表征，而非分类时拒绝。
3. **Webly监督学习**：Chen & Gupta [15]、Sun等[62]利用网络爬取的 noisy labeled 数据训练。本文不使用任何标签信息，仅利用无标注开放集的结构化采样。
4. **开放集半监督学习（OpenSemi）**：Saito等[56]、Su等[61]在含未知类的训练集上训练。本文框架更轻量，无需训练时已知标签集合划分。
5. **主动学习中的Coreset选择**：Sener & Savarese [58]、Wei等[72]选择最具代表性的标注样本。本文将coreset思想应用于**无标注**开放集的语义对齐。
6. **SSL中的Hard Negative Mining**：Robinson等[55]、Wang & Liu[70]利用困难负样本改进对比学习。本文的coreset本质上是选择与目标集"最相似"的正向样本，视角不同。

## 局限性与未来方向
1. **开放集质量依赖**：虽然SimCore对uncurated开放集鲁棒，但开放集本身需提供足够的语义覆盖（如Places365对Pet有意外收获，但对鸟类帮助有限）。
2. **k-means聚类假设**：使用目标集质心替代全部样本会损失簇内细节，当目标集极小或簇分布极不均匀时可能影响采样质量。
3. **停止准则阈值固定**：$\tau=0.95$ 为经验值，对不同分布差异程度的数据集可能非最优。
4. **计算开销**：需先用小规模数据预训练编码器以获取特征，再进行coreset采样，增加预训练管线复杂度。
5. **未来方向**：可扩展至多模态开放集（文本+图像）、动态自适应停止阈值设计、以及与其他数据筛选方法的结合。

## 研究启发与可借鉴点
1. **Coreset思想迁移**：将coreset采样应用于开放集自监督学习是一个新思路，可迁移到领域适应、跨域迁移等场景。
2. **隐空间相似度作为筛选标准**：使用预训练编码器的特征余弦相似度衡量样本相关性，简单且有效，可推广到其他自监督数据选择任务。
3. **自适应采样预算**：停止准则的比值机制可作为一种通用的"质量衰减监测"工具，用于其他迭代采样或数据筛选过程。
4. **实验设计的完备性**：本文在11数据集、7开放集、多种架构和SSL方法上验证，实验全面，值得借鉴其系统化评估策略。
5. **与下游任务的联合验证**：不仅评估线性探测，还验证了检测、分割、半监督等任务，充分展示了预训练表征的通用性。

## 关键术语表
**OpenSSL（Open-Set Self-Supervised Learning）**：在细粒度目标数据集预训练时，同时利用大规模无标注开放集进行自监督学习的新型问题设定。
**Coreset**：从大规模数据集中选取的具有代表性的子集，能够在保留原始数据分布特性的前提下大幅降低计算开销。
**SimCore**：本文提出的简单高效的coreset采样算法，通过目标集质心在隐空间中匹配最近的开放集样本。
**停止准则（Stopping Criterion）**：基于相邻采样轮次目标相似度比值判断是否继续采样的机制，防止无关样本污染。
**线性评估（Linear Evaluation）**：冻结预训练编码器，仅在下游任务上训练线性分类器以评估表征质量的协议。
**k-means聚类优化**：用目标集聚类质心替代全部目标样本进行相似度计算，将复杂度从$\mathcal{O}(|X| \cdot |\mathcal{U}|)$降至$\mathcal{O}(|\hat{X}| \cdot |\mathcal{U}|)$。

## 可复现要素
- **数据集**：11个细粒度数据集（Aircraft、Cars、Pet、Birds、Dogs、Flowers、Action、Indoor、Textures、Faces、Food）均为公开基准；开放集包括ImageNet-1k、MS COCO、iNaturalist 2021-mini、Places365、WebVision、WebFG-496，均已公开。
- **代码/权重**：论文未提及代码开源链接（CVPR 2023 paper，需进一步确认补充材料或项目页面）。
- **关键超参**：k-means质心数 $k=100$；停止阈值 $\tau = 0.95$；采样预算 $p=1\%$ 和 $5\%$；默认编码器为ResNet50，SSL方法为SimCLR。
