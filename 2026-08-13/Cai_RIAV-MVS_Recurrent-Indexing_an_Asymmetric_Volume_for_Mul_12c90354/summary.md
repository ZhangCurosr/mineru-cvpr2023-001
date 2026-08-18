---
title: "RIAV-MVS: Recurrent-Indexing an Asymmetric Volume for Multi-View Stereo"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Cai_RIAV-MVS_Recurrent-Indexing_an_Asymmetric_Volume_for_Multi-View_Stereo_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:42:17"
---

# 论文速读：RIAV-MVS: Recurrent-Indexing an Asymmetric Volume for Multi-View Stereo

## 一句话总结
提出一种基于“学习-优化”范式的端到端多视角立体视觉（MVS）深度估计方法，通过卷积GRU迭代预测连续索引场（Index Field）动态检索平面扫描代价体，并结合仅在参考视图生效的Transformer非对称特征增强与残差位姿校正网络，在ScanNet、DTU及跨域零样本测试上均取得SOTA精度。

## 研究问题与动机
- **2D CNN MVS削弱了几何先验**：现有方法（如Deep-VideoMVS）依赖跳跃连接融合多尺度特征进行深度解码，牺牲了代价体中嵌入的严格多视图几何约束，导致在未见域上泛化能力下降。
- **3D CNN的soft-argmin存在多模态退化**：soft-argmin只能输出代价体分布的期望值，在纹理缺失、重复纹理或遮挡导致的深度多峰分布区域会产生模糊或错误平均，无法定位最优候选深度。
- **传统迭代匹配缺乏可微性**：SGM等经典算法的winner-take-all操作不可微，其可微变体仍需预定义搜索方向且难以处理复杂分布，缺乏一种能锚定在代价体域内、全程梯度通路的迭代优化机制。
- **SLAM初始位姿含噪影响匹配质量**：实际采集序列的相机位姿通常由Visual SLAM提供，不可避免地含有平移/旋转误差，未经校正的位姿会导致平面扫描时的反向投影对齐偏差，直接劣化代价体构建。

## 核心贡献（创新点）
- **GRU
