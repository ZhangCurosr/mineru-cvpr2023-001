---
title: "Connecting-the-Dots-Floorplan-Reconstruction-Using-Two-Level"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Yue_Connecting_the_Dots_Floorplan_Reconstruction_Using_Two-Level_Queries_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:44:13"
field: "室内场景重建"
keywords: ["floorplan reconstruction", "two-level queries", "polygon prediction", "transformer", "structured prediction", "3D point cloud", "end-to-end training"]
innovations: ["提出两层查询 Transformer 架构实现可变顶点多边形并行预测", "设计多边形匹配策略处理循环等价性和可变长度序列", "单阶段端到端可训练避免多阶段启发式设计"]
benchmarks: ["Structured3D", "SceneCAD"]
---

# 论文速读：Connecting-the-Dots-Floorplan-Reconstruction-Using-Two-Level

## 一句话总结
本文提出 RoomFormer，一种基于 Transformer 的单阶段端到端可训练架构，通过两层查询（room 级和 corner 级）直接从 3D 点云俯视密度图并行生成多个房间的多边形（有序角点序列），在 Structured3D 和 SceneCAD 两个基准上均达到 SOTA，且推理速度比已有方法快 10 倍以上。

## 研究问题与动机
1. **多阶段流水线依赖手工设计**：现有 top-down 方法（如 Floor-SP、MonteFloor）先用 Mask R-CNN 检测房间掩码，再用整数规划/MCTS 提取多边形；bottom-up 方法（如 FloorNet、HEAT）先检测角点再连接边后组装平面图，均依赖启发式中间步骤且非端到端可训练。
2. **错误传播严重**：多阶段方法第二阶段的输入是第一阶段检测结果，初始角点或房间漏检/误检会直接影响最终重建质量。
3. **现有 Transformer 结构化重建方法仅支持固定参数形状**：DETR、LETR、PlaneTR 等预测边界框、线段、平面等固定参数形状，而楼层图多边形具有可变数量和有序顶点，无法直接应用。
4. **需要统一的全局建模**：楼层图重建需同时建模房间间关系和房间内部几何结构，现有方法缺乏全局推理能力。

## 核心贡献（创新点）
1. **重新表述楼层图重建为单阶段结构化预测问题**：将问题定义为直接预测可变数量房间的有序角点序列集合，避免多阶段启发式设计；与已有方法相比，无需显式角点检测、边分类或后处理优化。
2. **提出两层查询的 Transformer 架构（RoomFormer）**：引入 room 级和 corner 级双层查询，支持并行预测可变数量房间和可变长度顶点序列；与 DETR/LETR 等仅支持固定参数形状的方法本质不同。
3. **设计多层级多边形匹配策略**：在 room 级通过匈牙利算法建立二分匹配，在 corner 级考虑多边形循环等价性（固定顺时针方向，枚举所有起始顶点取最小距离）；相比 HEAT 等依赖角点-边匹配的方法，实现真正的端到端训练。
4. **灵活扩展至语义丰富楼层图**：仅需少量架构调整即可同时预测房间类型、门、窗等元素；相比需额外模块的方法，扩展更为简洁统一。

## 方法详解
1. **输入表示**：将 3D 点云沿重力轴投影为 256×256 俯视密度图，像素值为投影点数除以最大点数，归一化到 [0, 1]。
2. **CNN 骨干网络**：使用 ResNet-50 提取最后三个 stage 的多尺度特征图（通道数降至 256），每尺度添加正弦/余弦位置编码后展平拼接。
3. **Transformer 编码器**：采用多尺度可变形注意力（multi-scale deformable attention），每个 query 仅关注 N_s=4 个采样点，降低计算复杂度并聚合多尺度上下文。
4. **两层查询设计**：
   - 查询张量 Q ∈ R^(M×N×2)，M=20 为房间级最大数量，N=40 为角点级最大数量。
   - 每个查询包含内容查询（decoder embeddings）和位置查询（由 polygon queries 经 MLP+PE 生成）。
   - Decoder 中所有角点级查询进行 self-attention，允许同一房间和相邻房间角点交互。
   - 多尺度可变形 cross-attention 以 polygon queries 为参考点，从多尺度特征图聚合特征。
5. **迭代多边形细化**：每层 decoder 输出 MLP 预测相对偏移 (Δx, Δy)，更新 polygon queries 供下一层使用，类似 Deformable DETR 的边界框细化策略。
6. **多边形匹配**：
   - 匈牙利算法在 room 级建立二分匹配，最小化代价函数。
   - 代价函数 = λ_cls·∑|c_n^m - ĉ_n^σ(m)| + λ_coord·d(P_m, P̂_σ(m))，其中 d 为不考虑 padding 的顶点对齐 L1 距离，考虑多边形循环等价性取最小值。
7. **损失函数**：
   - L_cls：顶点有效性二元交叉熵（所有查询）
   - L_coord：顶点坐标 L1 距离（仅匹配成功的多边形）
   - L_ras：多边形光栅化后 Dice 损失（辅助损失，使用可微光栅化器）
   - 总损失 L = Σ(2·L_cls + 5·L_coord + 1·L_ras)

## 实验与结果
1. **数据集**：
   - Structured3D：3500 栋房屋（3000/250/250 训练/验证/测试），16 种房间类型，含门、窗标注，密度图尺寸 256×256
   - SceneCAD：基于 ScanNet 真实 RGB-D 扫描，训练/验证集共用，密度图同尺寸
2. **评估指标**：房间级 IoU/精确率/召回率/F1；角点级精确率/召回率/F1；角度级精确率/召回率/F1
3. **Structured3D 测试结果**（表 1）：
   - 房间 F1：97.3%（较 HEAT 的 95.4% 提升 **+1.9**）
   - 角点 F1：87.2%（较 HEAT 的 82.5% 提升 **+4.7**）
   - 角度 F1：81.2%（较 HEAT 的 78.3% 提升 **+2.9**）
   - 推理速度：0.01s（HEAT 为 0.11s，**快 10 倍+**）
4. **SceneCAD 验证集结果**（表 2）：
   - 房间 IoU：91.7%（vs Floor-SP 91.6%，HEAT 84.9%）
   - 角点 F1：88.8%（vs HEAT 83.2%）
   - 推理速度：0.01s（HEAT 为 0.12s）
5. **跨域泛化**（表 3）：Structured3D 训练→SceneCAD 验证，房间 IoU 74.0%（vs HEAT 52.5%，**提升 21.5%**），证明全局建模优势。
6. **语义增强结果**（表 4）：SD-TQ 变体门/窗 F1 达 81.1%，房间类型 F1 达 70.7%。

## 相关工作脉络
1. **Top-down 方法（Floor-SP [8], MonteFloor [29]）**：先 Mask R-CNN 检测房间掩码，再整数规划/MCTS 提取多边形；本文直接预测多边形，无需掩码分割和启发式搜索。
2. **Bottom-up 方法（FloorNet [21], HEAT [9]）**：先检测角点，再分类边，最后组装平面图；本文跳过显式角点/边检测，直接生成完整多边形序列。
3. **通用线段检测（HAWP [35], LETR [34]）**：检测线条后适配楼层图；本文专为可变顶点多边形设计，而非固定参数形状。
4. **Transformer 结构化重建（DETR [7], PlaneTR [30]）**：预测边界框/平面等固定参数形状；本文处理有序顶点序列，支持可变长度和循环等价性。
5. **实例分割多边形预测（Polytransform [19], [18]）**：假设固定顶点数或使用边界框初始化；本文无需固定顶点数和边界框初始化，直接并行生成多个多边形。

## 局限性与未来方向
1. 模型假设多边形为简单闭合形状，对含孔洞或复杂连通性的房间重建能力有限。
2. 仅使用俯视密度图，未充分利用 3D 几何信息（如高度、墙体厚度）。
3. 门/窗预测需增加查询数量，对大量小型元素的场景效率可能下降。
4. 未来可探索扩展至 3D 楼层图重建，或结合语义分割进一步提升精度。

## 研究启发与可借鉴点
1. **两层查询设计**：对于需预测有序序列的结构化输出任务（如多边形、轨迹、点云配准），可借鉴外层实例级+内层元素级的双层查询架构。
2. **循环等价性处理**：多边形匹配中固定方向后枚举所有起始顶点取最小距离的策略，可迁移至其他有序环状序列预测任务。
3. **迭代细化机制**：decoder 层间更新坐标查询作为下一层可变形注意力参考点，该策略可从目标检测迁移到其他结构化重建任务。
4. **多尺度可变形注意力**：结合多尺度特征与可变形注意力，兼顾局部细节和全局上下文，适用于需精确定位的视觉任务。
5. **统一多任务框架**：通过调整查询数量和类型，在同一架构下预测多种输出（房间、角点、门、窗、类型），为多任务学习提供简洁方案。

## 关键术语表
1. **Two-Level Queries**：双层查询设计，一层表示房间（polygon）实例，另一层表示每个房间内的角点（corner）序列，用于处理可变数量实例和可变长度序列。
2. **Deformable Attention**：可变形注意力，query 仅关注参考点周围的少量采样点（N_s=4），降低计算复杂度并聚焦局部区域。
3. **Polygon Matching**：多边形匹配策略，通过匈牙利算法在 room 级建立二分匹配，结合顶点标签和坐标计算匹配代价，考虑多边形循环等价性。
4. **Density Map**：密度图，将 3D 点云沿重力轴投影到 2D 平面形成的像素密度图像，突出建筑结构元素（如墙壁）。
5. **Iterative Refinement**：迭代细化，decoder 每层预测相对于当前查询位置的偏移量，逐步精炼多边形顶点的预测位置。
6. **End-to-End Trainable**：端到端可训练，整个模型从输入密度图到输出多边形集合可直接通过反向传播优化，无需人工设计中间模块。
7. **Single-Stage Prediction**：单阶段预测，无需多阶段流水线（如先检测角点再连接边），直接并行生成所有房间的多边形。

## 可复现要素
1. **数据集**：
   - Structured3D：公开数据集，https://structured3d-dataset.org/
   - SceneCAD：基于 ScanNet，需在 ScanNet 数据集基础上转换（https://kaldir.vc.in.tum.de/scenedcad/）
2. **代码开源**：是，https://github.com/ywyue/RoomFormer
3. **模型权重**：是，论文提供预训练权重
4. **关键超参**：
   - M=20（房间级查询数量），N=40（角点级查询数量）
   - Transformer：6 层 encoder，6 层 decoder，256 通道，8 heads
   - 可变形注意力采样点数 N_s=4
   - 损失系数：λ_cls=2, λ_coord=5, λ_ras=1
   - 优化器：Adam，weight decay=1e-4
   - 学习率：Structured3D 上 2e-4，SceneCAD 上 5e-5
   - 训练轮数：Structured3D 500 epochs，SceneCAD 400 epochs
