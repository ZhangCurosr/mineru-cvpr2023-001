---
title: "Connecting-the-Dots-Floorplan-Reconstruction-Using-Two-Level"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Yue_Connecting_the_Dots_Floorplan_Reconstruction_Using_Two-Level_Queries_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:44:28"
---

# 论文速读：Connecting-the-Dots-Floorplan-Reconstruction-Using-Two-Level

## 一句话总结
本文提出 RoomFormer，将 2D 户型图重建形式化为单一阶段的端到端集合预测任务，通过引入房间级与角点级两级查询（Two-Level Queries）及多边形匈牙利匹配策略，直接从 3D 点云密度图中并行预测可变数量的房间多边形及其有序顶点序列，在 Structured3D 与 SceneCAD 上均刷新 SOTA，且推理速度较 prior works 提升 10 倍以上。

## 研究问题与动机
- 现有方法多为启发式多阶段流水线：自上而下（如 Floor-SP、MonteFloor）先分割房间掩码再用整数规划或蒙特卡洛搜索提取多边形，非端到端且高度依赖第一阶段分割质量。
- 自下而上方法（如 HEAT、FloorNet）先检测角点再分类边或组装图结构，阶段间误差易累积，且在点云稀疏时极易丢失角点/边。
- 现有 Transformer 结构化重建工作（DETR、LETR、PlaneTR 等）主要针对固定参数几何基元（包围盒、直线、平面），无法原生处理顶点数量任意且顺序敏感的闭合多边形。
- 传统管线普遍依赖后处理（如 Douglas-Peucker 简化、非极大值抑制、手工规则约束），推理慢且缺乏全局一致性推理能力。

## 核心贡献（创新点）
1. **端到端单阶段建模**：将户型图重建重新定义为同时生成多个有序顶点序列的集合预测问题，彻底摒弃手工中间表示与后处理步骤。
   *本质区别*：区别于前作分步检测角点/墙/房间，本文直接由密度图映射至完整多边形集合，消除阶段误差传递。
2. **两级查询架构**：设计 $M \times N$ 查询矩阵，上层决策房间是否存在（valid/invalid），下层预测每个房间的顺序顶点坐标，通过有效性分类自适应容纳变长序列。
   *本质区别*：与 DETR/LETR 等固定参数或固定顶点假设不同，本文通过查询解耦与分类机制原生支持任意房间数与任意顶点数。
3. **多边形匹配策略**：提出结合匈牙利算法的双层匹配机制，在集合层匹配房间，在序列层匹配顶点（对闭合多边形的所有循环平移取最小代价），实现端到端可微训练。
   *本质区别*：解决了变长有序序列与固定预测头之间的对齐难题，避免了 prior works 中依赖贪心规则或显式优化的匹配方式。
4. **语义与结构无缝扩展**：利用已学习的房间级聚合特征直接预测房间类型，并将门窗视为 2 顶点特殊多边形复用主解码器，保持单阶段架构完整性。
   *本质区别*：现有方法通常需额外独立分支或后处理模块添加语义，本文在统一框架内实现几何+语义联合预测。

## 方法详解
- **输入与骨干网络**：将 3D 点云沿重力轴投影生成 256×256 归一化密度图。采用 ResNet-50 提取最后三个 stage 的多尺度特征，经 1×1 卷积统一至 256 维，展平后叠加正弦/余弦位置编码，送入 Transformer Encoder。
- **多级可变形注意力**：Encoder/Decoder 均采用 multi-scale deformable attention，每个查询仅关注特征图上 $N_s=4$ 个参考点附近的采样区域，兼顾局部细节与全局上下文，避免标准自注意力的 $O((HW)^2)$ 复杂度。
- **两级查询与迭代
