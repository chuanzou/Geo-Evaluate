# 三方法对比分析：temporal vs GAN vs diffusion

对应图：`outputs/figures/all_methods_curves.png`
数据：`outputs/metrics/{temporal,gan,diffusion}_metrics_by_coverage.csv`
云量区间：0.05 / 0.10 / 0.30 / 0.50 / 0.70（每 bin 600 个 test 样本）

## 核心指标汇总

| coverage | method    | PSNR   | SSIM   | L1     | L1_cloud | NDVI_MAE |
|:--------:|:----------|:------:|:------:|:------:|:--------:|:--------:|
| 0.05     | temporal  | ≈175   | ≈1.00  | ≈0     | ≈0       | ≈0       |
| 0.05     | gan       | 43.44  | 0.988  | 0.0011 | 0.0222   | 0.0995   |
| 0.05     | diffusion | 40.83  | 0.975  | 0.0016 | 0.0310   | 0.1513   |
| 0.30     | temporal  | ≈25    | 0.36   | 0.01   | 0.02     | 0.015    |
| 0.30     | gan       | 35.25  | 0.925  | 0.0068 | 0.0228   | 0.1039   |
| 0.30     | diffusion | 31.86  | 0.838  | 0.0104 | 0.0347   | 0.1701   |
| 0.70     | temporal  | ≈5     | 0.01   | 0.21   | 0.30     | 0.18     |
| 0.70     | gan       | 30.49  | 0.768  | 0.0182 | 0.0260   | 0.1234   |
| 0.70     | diffusion | 26.82  | 0.581  | 0.0272 | 0.0403   | 0.2095   |

（temporal 的低云量 PSNR 本应无穷大，受浮点精度约 ~175 dB）

## 观察 1：三种方法各有胜区，交叉点在 coverage ≈ 0.25–0.30

`L1_cloud_region` 曲线最直观。GAN 和 diffusion 的 `L1_cloud` 在整个云量范围内几乎是平的（GAN 0.022→0.026，diffusion 0.031→0.040），temporal 从 0 飙到 0.30。

- **coverage < 0.30**：temporal 基本完美，generative 方法反而多余。
- **coverage > 0.30**：temporal 急剧退化，generative 稳定。

这就是这项工作要回答的 trade-off —— **数据冗余 vs. 单帧建模能力**。低云量下不需要复杂模型，把多帧拼起来就够；高云量下多帧也没救，只能靠学到的先验去补。

## 观察 2：temporal 在 coverage=0.05 的 PSNR ≈ 175 dB 不是 bug

评估时每张 test 图额外合成了 `temporal_neighbors=3` 张 cloud-mask 不同的观测，加上原图共 4 帧。某个像素在全部 4 帧里都被云覆盖的概率是 `coverage^4`：

| coverage | P(all frames covered) | 期望"全遮"像素数 (128×128) |
|:--------:|:---------------------:|:-------------------------:|
| 0.05     | 6.25×10⁻⁶             | ~0.1                      |
| 0.10     | 1.00×10⁻⁴             | ~1.6                      |
| 0.30     | 8.10×10⁻³             | ~133                      |
| 0.50     | 6.25×10⁻²             | ~1024                     |
| 0.70     | 2.40×10⁻¹             | ~3932                     |

低云量下 temporal 相当于"直接读 target"，MSE 趋于浮点精度极限，PSNR 自然爆高；SSIM≈1、L1≈0、NDVI_MAE≈0 一起印证。

## 观察 3：temporal 的退化曲线由 "P(all covered)" 驱动，是 coverage 的 N 次方

N=4 的幂律让 temporal 在 0.5 → 0.7 区间几乎线性跳崖（因为 `cov^4` 从 0.06 跳到 0.24）。而 GAN/diffusion 的瓶颈是"单帧上下文能推多远"，跟云量是弱相关关系，所以曲线平。

这也暗示：如果把 `temporal_neighbors` 从 3 改成 5 或 8，temporal 的"拐点"会右移到更高 coverage。当前 3 张邻居的设定对 high-coverage 区间不友好，可以作为后续消融的一个自然轴。

## 观察 4：GAN 全面优于 diffusion，但差距稳定在可解释范围内

| coverage | PSNR 差距 (dB) | L1_cloud 差距倍数 |
|:--------:|:--------------:|:-----------------:|
| 0.05     | 2.6            | 1.40×             |
| 0.10     | 2.8            | 1.48×             |
| 0.30     | 3.4            | 1.52×             |
| 0.50     | 3.6            | 1.55×             |
| 0.70     | 3.7            | 1.55×             |

差距随 coverage 略微变大，但非常稳定（PSNR 永远差 3±1 dB，L1_cloud 永远差 1.4–1.6×）。这种"同形状平移"说明两者都已经进入各自的能力上限，没有爆出采样失败、模式坍缩之类的病态现象。

理由侧：
- 两者共用同一个 0.08M 参数小 UNet。
- GAN 有 L1 loss + 对抗 loss **直接监督 target**。
- Diffusion 只通过 ε-prediction **间接监督**，信号多绕一层，训练效率天然低一个档次。
- T=100 linear schedule，α̅_T ≈ 0.37，靠 mask 外均值做 init（见 `diffusion_init_bias_fix.md`）是能用的最好近似，但仍有细小 distribution shift。

## 观察 5：diffusion 的 NDVI_MAE 在所有 coverage 下都最差 —— 这是"通道协方差"问题

这是唯一一个 diffusion 在任何条件下都不如 baseline 的指标：

| coverage | temporal | gan   | diffusion | diffusion / gan |
|:--------:|:--------:|:-----:|:---------:|:---------------:|
| 0.05     | ≈0       | 0.0995| 0.1513    | 1.52×           |
| 0.10     | ≈0       | 0.0994| 0.1554    | 1.56×           |
| 0.30     | 0.015    | 0.1039| 0.1701    | 1.64×           |
| 0.50     | 0.066    | 0.1108| 0.1844    | 1.66×           |
| 0.70     | 0.18     | 0.1234| 0.2095    | 1.70×           |

注意：PSNR/L1 上 diffusion 只比 GAN 差 1.4×，但 NDVI_MAE 差 1.5–1.7×。**不均匀的劣势说明问题不是"所有通道都错一点"，而是"通道之间比例错得更多"**。

NDVI = (NIR − R) / (NIR + R + ε)。这个指标对每个通道的**绝对**值容差大，但对 NIR / R 的**比例**容差很小。Diffusion 在 4 个通道上独立做 ε-prediction，训练目标里没有任何约束要求预测的 ε 在通道间保持和真实 ε 一样的相关结构；GAN 的 L1 loss 作用在整个 4 通道向量上、再过一个 PatchGAN discriminator，对联合分布多少有点约束，因此通道间比例更准。

## 可能的改进方向

按投入产出从高到低：

1. **给 diffusion 加 NDVI 一致性辅助 loss**：训练时同时惩罚 `|NDVI(pred_x0) − NDVI(target)|`。单行 loss，能精准打在这条最弱的指标上，不影响 PSNR/L1。
2. **改 x0-parameterization**：让 denoiser 直接预测 target 而非 ε；与 L1-loss 的距离更近，通道间联合分布更容易学到。
3. **cosine schedule 或 T=1000**：让 α̅_T → 0，init 从纯噪声开始变得 in-distribution，可以进一步缩小 diffusion 和 GAN 的总体差距。
4. **`temporal_neighbors` 消融**：把 N 从 3 扩到 5、8、12，画"交叉点 vs N"的曲线，这能让 temporal baseline 的优势区间系统化地被刻画出来，是 report 里一条很好讲的 story。
5. **放大 denoiser**：`base_channels=32` 对 GAN 够用（有直接监督），对 diffusion 偏紧。翻倍到 64 大概率能再咬下 1–2 dB 的差距。

## report 里可以直接用的一句话结论

> 多时相融合在低云量（<30%）下接近完美但高云量下以 `cov^N` 速率退化；生成式方法对云量几乎不敏感，其中 GAN 全面优于同架构的小型 diffusion（PSNR 差 3±1 dB），而 diffusion 在植被指数保持（NDVI）上表现最弱，提示 ε-prediction 目标下的通道间协方差建模是该 baseline 的主要失分点。
