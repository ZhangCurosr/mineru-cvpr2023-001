---
title: "BAD-NeRF-Bundle-Adjusted-Deblur-Neural-Radiance-Fields"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Wang_BAD-NeRF_Bundle_Adjusted_Deblur_Neural_Radiance_Fields_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:51:26"
field: "NeRF 与 3D 重建"
keywords: ["NeRF", "运动模糊", "Bundle Adjustment", "Novel View Synthesis", "Deblurring", "Camera Pose Estimation", "SE(3)"]
innovations: ["提出光度束调整框架，显式建模运动模糊物理成像过程，联合优化 NeRF 与曝光时间内相机运动轨迹", "使用 SE(3) 线性插值表示曝光轨迹，区别于固定姿态优化方法", "通过平均虚拟清晰图像替代 PSF 卷积，自然建模遮挡关系"]
benchmarks: ["Deblur-NeRF Synthetic Dataset", "MBA-VO Dataset"]
---

# 论文速读：BAD-NeRF: Bundle Adjusted Deblur Neural Radiance Fields

## 一句话总结
BAD-NeRF 提出了一种光度束调整框架，通过显式建模运动模糊图像的物理成像过程，在严重运动模糊和 inaccurate 初始相机姿态下，联合优化 NeRF 场景表示与曝光时间内相机运动轨迹，实现高质量的去模糊与 novel view 合成。

## 研究问题与动机
- **NeRF 假设输入为清晰图像**：标准 NeRF 假设渲染图像对应无限短曝光时间，运动模糊图像破坏了这一假设，导致训练退化。
- **相机姿态难以从模糊图像中精确估计**：严重运动模糊使关键点对检与匹配困难，COLMAP 等常规方法得到的初始姿态往往不准确；而 Deblur-NeRF 依赖固定姿态（GT 或 COLMAP），对姿态误差极为敏感。
- **去模糊与 3D 重建耦合挑战**：传统单图去模糊方法（如 MPR、SRNDeblur）在严重模糊下失效；已有方法（Park et al.）采用稠密深度图表示场景，无法充分发挥 NeRF 的多视图一致性与 novel view 合成能力。
- **曝光时间内相机运动建模缺失**：现有方法通常优化固定时间戳的姿态，未建模曝光期间连续运动轨迹，限制了在低光长曝光条件下的重建鲁棒性。

## 核心贡献（创新点）
- **光度束调整形式化（Photometric Bundle Adjustment）**：将运动模糊图像的物理成像过程嵌入 NeRF 训练，联合优化 NeRF 权重与相机运动轨迹，区别于 Deblur-NeRF 固定姿态的做法，实现了姿态与场景表示的端到端协同学习。
- **SE(3) 空间线性轨迹建模**：用曝光起止两个位姿参数化运动轨迹，中间位姿在 SE(3) 李代数上线性插值，区别于 BARF/Jeong et al. 仅优化单一时刻姿态的方法。
- **物理成像模型的离散化合成**：沿运动轨迹渲染多张虚拟清晰图像并平均得到模糊图像，显式模拟真实曝光过程，区别于 Deblur-NeRF 使用卷积 PSF 的模糊建模方式（后者不建模遮挡）。
- **对 inaccurate 初始姿态的鲁棒性**：实验表明 BAD-NeRF 使用 COLMAP 初始姿态即可取得优异结果，而 Deblur-NeRF 在相同条件下性能显著下降（约 3-4 dB PSNR 损失），证明了联合优化的必要性。

## 方法详解
- **NeRF 基础表示**：沿用原始 NeRF 架构，使用两个 MLP（$F_\theta$）隐式表示 3D 场景，输入 5D 向量（空间位置 + 视角方向），输出颜色 $c$ 与体密度 $\sigma$；采用 Fourier embedding 提升表达能力。
- **可微体积渲染**：沿射线采样 3D 点，通过公式 $I(x) = \sum_{i=1}^{n} T_i(1-\exp(-\sigma_i \delta_i))c_i$ 计算像素强度，其中 $T_i$ 为透射率，整体对 $\theta$ 和相机姿态 $\mathbf{T}_c^w$ 可微。
- **运动模糊成像模型**：模糊图像 $B(x) = \phi \int_0^\tau I_t(x) dt$，离散近似为 $B(x) \approx \frac{1}{n}\sum_{i=0}^{n-1} I_i(x)$，即沿曝光时间渲染 $n$ 张虚拟清晰图像后平均。
- **相机运动轨迹建模**：曝光时间内用 $T_{start}$ 和 $T_{end}$ 两个 SE(3) 位姿表示，中间位姿通过 $T_t = T_{start} \cdot \exp(\frac{t}{\tau} \cdot \log(T_{start}^{-1} \cdot T_{end}))$ 线性插值得到；对短曝光时间（≤200ms），线性插值已足够精确。
- **联合优化损失函数**：最小化合成模糊图像与真实模糊图像的光度差 $\mathcal{L} = \sum_{k=0}^{K-1} \|B_k(x) - B_k^{gt}(x)\|$；通过链式法则计算对 $\theta$、$T_{start}$、$T_{end}$ 的梯度，分别用独立 Adam 优化器更新。
- **超参数设置**：曝光时间内插值虚拟相机数 $n=7$，每射线采样 128 点，训练 200K 次迭代；学习率指数衰减（NeRF 参数：$5\times10^{-4}\to5\times10^{-5}$，位姿参数：$1\times10^{-3}\to1\times10^{-5}$）；初始位姿由 COLMAP 提供。

## 实验与结果
- **数据集**：Deblur-NeRF 合成数据集（Blender 生成，5 个场景：Cozy2room、Factory、Pool、Tanabata、Trolley）与 MBA-VO 数据集（Unreal 引擎合成 + 真实手持相机采集，含 ArchViz-low/high 序列）。
- **评估基线**：SRNDeblurNet、PVD、MPR、Deblur-NeRF（及 Deblur-NeRF*，使用 GT 姿态）、Park et al.；评估指标为 PSNR、SSIM、LPIPS，以及 ATE（相机姿态估计）。
- **去模糊性能（Deblur-NeRF 合成集）**：BAD-NeRF 平均 PSNR 达 **30.94 dB**，较 Deblur-NeRF（25.56 dB）提升 **+5.38 dB**；较最强单图去模糊基线 MPR（27.42 dB）提升 **+3.52 dB**。在 Factory 高模糊场景中，BAD-NeRF 达 **32.08 dB**，超越 Deblur-NeRF*（GT 姿态，26.40 dB）**+5.68 dB**，展示了显著优势。
- **Novel View 合成性能**：BAD-NeRF 平均 PSNR 达 **29.28 dB**，优于 Deblur-NeRF（25.68 dB）**+3.60 dB**；在 Cozy2room 场景达 30.97 dB（SSIM 0.9014，LPIPS 0.0552），优于所有基线。
- **MBA-VO 数据集（非恒定速度运动）**：BAD-NeRF 在 ArchViz-low（PSNR 31.27 dB）和 ArchViz-high（PSNR 28.07 dB）上均取得最佳结果，证明线性模型对非匀速运动仍有效。
- **轨迹表示消融**：直接优化 7 个独立姿态（29.99/28.86 dB）显著差于线性插值（30.94/29.67 dB）和立方 B-Spline（30.89/29.93 dB），验证了轨迹约束的重要性；线性插值与 B-Spline 性能相近。
- **相机姿态估计（ATE）**：BAD-NeRF 在 Cozy2room、Factory、Pool、Tanabata、Trolley 五个序列上 ATE 分别为 0.050、0.033、0.020、0.016、0.007，显著优于 COLMAP-blur 和 BARF，证明联合优化可精确恢复曝光时间内轨迹。
- **最强结果总结**：Deblur-NeRF 合成集去模糊平均 PSNR **30.94 dB / SSIM 0.8946 / LPIPS 0.0916**；novel view 合成平均 **29.28 dB / SSIM 0.8734 / LPIPS 0.1045**；MBA-VO ArchViz-low **31.27 dB / SSIM 0.9005**；显著超越所有基线。

## 相关工作脉络
- **BARF（Chen-Hsuan Lin et al., ICCV 2021）**：联合优化 NeRF 与相机外参姿态，但优化的是固定时间戳的单一姿态；BAD-NeRF 进一步优化曝光时间内的连续运动轨迹。
- **Jeong et al.（ICCV 2021, Self-Calibrating NeRF）**：同时学习场景与相机内外参；BAD-NeRF 关注的是曝光时间内的运动轨迹而非标定参数。
- **Deblur-NeRF（Ma et al., CVPR 2022）**：首个面向模糊图像的 NeRF 方法，但依赖固定姿态且使用 PSF 卷积建模模糊（不建模遮挡）；BAD-NeRF 通过物理成像模型显式合成模糊，解决遮挡问题。
- **Park et al.（ICCV 2017）**：经典多视图去模糊方法，联合估计姿态、深度图与清晰图像；BAD-NeRF 以 NeRF 隐式替代稠密深度图，保留了 novel view 合成能力。
- **MBA-VO（Liu et al., ICCV 2021）**：运动模糊感知视觉里程计；BAD-NeRF 可与其结合，构成端到端的模糊图像 3D 重建管线。
- **单图/视频去模糊网络（MPR、PVD、SRNDeblur）**：基于深度学习的去模糊方法；BAD-NeRF 利用多视图一致性信息，避免了单图方法的 artifacts 问题，且直接支持 novel view 合成。

## 局限性与未来方向
- **线性运动模型限制**：假设曝光时间内相机匀速运动，对于更长曝光时间或复杂加速度运动可能不够准确（论文提及 cubic B-Spline 可作为替代，但线性已足够）。
- **计算开销**：需在曝光时间内渲染多张虚拟清晰图像（实验中 n=7），训练时间较长（200K 迭代）。
- **曝光时间未知**：当前方法假设已知曝光时间 τ，实际应用中 τ 可能需要联合估计。
- **动态场景扩展**：当前方法针对静态场景，未来可探索与动态 NeRF（如 D-NeRF、NeRFies）结合处理含运动物体的场景。
- **Real-to-Sim 差距**：合成数据集假设恒定速度运动，真实数据集中存在非匀速运动，虽然实验表明效果良好，但在极端情况下可能受限。

## 研究启发与可借鉴点
- **物理成像模型的嵌入策略**：将退化过程（运动模糊）以可微分的物理模型直接嵌入 NeRF 训练框架，而非先处理再训练的两阶段范式，这一思路可推广至其他退化类型（如散焦模糊、低光照噪声）。
- **SE(3) 线性插值 + 联合优化**：使用李代数参数化轨迹并在曝光区间内线性插值，是处理短曝光运动的优雅方案，可迁移到 SLAM/VO 与 NeRF 结合的场景。
- **遮挡建模的显式优势**：通过平均虚拟清晰图像而非卷积 PSF 的方式，自然处理了深度不连续处的遮挡问题，为后续去模糊 NeRF 工作提供了重要设计参考。
- **姿态敏感性分析范式**：论文通过 Deblur-NeRF*（GT 姿态）与 Deblur-NeRF（COLMAP 姿态）的对比，清晰展示了方法对初始姿态误差的鲁棒性，这一实验设计值得借鉴。
- **可与实时 NeRF 结合**：BAD-NeRF 框架与 Instant NeRF / PlenOctrees 等加速方法兼容，未来可探索实时模糊场景重建。

## 关键术语表
- **NeRF（Neural Radiance Fields）**：使用 MLP 隐式表示 3D 场景的神经渲染方法，输入空间坐标和视角方向，输出颜色与体密度，通过可微体积渲染生成新视角图像。
- **Bundle Adjustment（束调整）**：同时优化场景参数与相机位姿的联合优化策略，通过最小化重投影误差实现自洽估计。
- **SE(3)**：三维刚体变换的李群，包含旋转和平移，其李代数可用于参数的微分和插值运算。
- **Photometric Loss（光度损失）**：基于像素强度差的损失函数，衡量合成图像与真实图像之间的一致性。
- **Volume Rendering（体积渲染）**：沿光线采样 3D 点并累加颜色和透明度以生成 2D 像素值的可微渲染技术。
- **FOV Embedding（Fourier Embedding）**：将低维坐标映射到高维周期空间的编码方式，提升 NeRF 对高频细节的学习能力。
- **ATE（Absolute Trajectory Error）**：视觉里程计中评估估计轨迹与真实轨迹对齐后绝对误差的常用指标。

## 可复现要素
- **数据集**：Deblur-NeRF 合成数据集（Blender 生成，5 场景）和 MBA-VO 数据集（Unreal + 真实采集）——代码仓库链接已公开（https://github.com/WU-CVGL/BAD-NeRF）；论文未提及额外自采集数据集。
- **代码/权重**：代码和权重已开源（https://github.com/WU-CVGL/BAD-NeRF）。
- **关键超参**：虚拟相机数 n=7，每射线采样点数 128，训练迭代数 200K，NeRF 学习率 $5\times10^{-4}\to5\times10^{-5}$，位姿学习率 $1\times10^{-3}\to1\times10^{-5}$，Adam 优化器，初始姿态由 COLMAP 提供。
