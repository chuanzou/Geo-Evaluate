# Diffusion 推理异常：根因与修复

## 观察到的问题

`outputs/metrics/diffusion_metrics_by_coverage.csv`：

| coverage | PSNR  | SSIM   | L1     | L1_cloud | ndvi_mae |
|---------:|------:|-------:|-------:|---------:|---------:|
| 0.05     | 19.59 | 0.1747 | 0.0199 | 0.3973   | 0.5225   |
| 0.10     | 15.57 | 0.0788 | 0.0460 | 0.4595   | 0.5492   |
| 0.30     |  9.42 | 0.0156 | 0.1698 | 0.5662   | 0.5789   |
| 0.50     |  6.56 | 0.0071 | 0.3118 | 0.6236   | 0.5839   |
| 0.70     |  4.64 | 0.0039 | 0.4677 | 0.6682   | 0.5840   |

对比 GAN 的 PSNR 30–43、L1_cloud 0.022–0.026，diffusion 明显没在学云区的真实内容。

## 关键线索

**L1 总误差几乎精确等于 L1_cloud × coverage**（验证 compositing 步骤没问题，错误全在云区）：

```
coverage × L1_cloud ≈ L1     (0.05·0.397=0.0199 ✓, 0.70·0.668=0.468 ✓)
```

**测试集内云区 target 像素的真实统计量**（用 `scripts/diagnose_no_torch.py` 跑了 80 × 5 个样本）：

| coverage | mean   | std    | L1(pred=0) | L1(pred=mean) | L1(pred=1) |
|---------:|-------:|-------:|-----------:|--------------:|-----------:|
| 0.05     | 0.1199 | 0.0968 | 0.1199     | 0.0805        | 0.8801     |
| 0.10     | 0.1226 | 0.0961 | 0.1226     | 0.0803        | 0.8774     |
| 0.30     | 0.1206 | 0.0944 | 0.1206     | 0.0783        | 0.8794     |
| 0.50     | 0.1235 | 0.0985 | 0.1235     | 0.0829        | 0.8765     |
| 0.70     | 0.1210 | 0.0968 | 0.1210     | 0.0811        | 0.8790     |

即：最优常值预测器应得 L1_cloud ≈ 0.08；预测纯白会得 0.88。

**diffusion 得到 0.40 → 0.67**，介于两者之间且随云量增大单调趋近于白。把它建模为

```
pred ≈ α · 1 + (1 − α) · target
L1   = α · (1 − target_mean) = α · 0.88
```

反推 α（白色偏置比例）：

| coverage | 0.05 | 0.10 | 0.30 | 0.50 | 0.70 |
|---------:|-----:|-----:|-----:|-----:|-----:|
| α        | 0.45 | 0.52 | 0.65 | 0.71 | 0.76 |

## 根因

`InpaintingDiffusion.inpaint()` 之前的初始化：

```python
x = sqrt(α̅_{T-1}) · cloudy  +  sqrt(1 − α̅_{T-1}) · noise
  ≈ 0.61 · cloudy + 0.79 · noise
```

训练时 denoiser 只看过 `x = 0.61·target + 0.79·noise`，这里 `target_mean ≈ 0.12`，
所以训练分布 `mean(x) ≈ 0.073`。

推理时把 `cloudy` 当 target 代入，云区 `cloudy ≈ 1`，`mean(x) ≈ 0.61`，
**mean 偏差 8×**。denoiser 把这个输入当作「很亮的真实场景」来反推噪声，
每一步都把 x 往「1」的方向拉，DDPM chain 没法自愈 → 最终输出偏白。

旁证：更早的 `diffusion_metrics_by_coverage_buggy.csv` 用的是纯 `randn` 初始化，
L1_cloud ≈ 0.271 稳定不变——那种 init 在 std 上 OOD 但 mean 接近 0，反而
比「0.61 cloudy 偏置」要好。

## 修复（不需重训）

把初始化里的 `cloudy` 换成**本图像 mask 外区域的均值**（逐 batch、逐通道）：

```python
clean_weight = (1.0 - mask)
clean_denom  = clean_weight.sum(dim=(-2,-1), keepdim=True).clamp(min=1.0)
clean_mean   = (cloudy * clean_weight).sum(dim=(-2,-1), keepdim=True) / clean_denom
target_proxy = clean_mean.expand_as(cloudy)

x = sqrt(α̅_{T-1}) · target_proxy + sqrt(1 − α̅_{T-1}) · noise
```

理由：合成数据下 mask 外 `cloudy == target`，`clean_mean` 就是这张图 target
像素均值的无偏估计；代进公式得到的 x 和训练分布的 mean/std 都匹配。

已改 `src/cloud_removal/models/diffusion.py::inpaint`。重跑：

```bash
bash scripts/rerun_diffusion_eval.sh
```

会覆盖 `outputs/metrics/diffusion_metrics.csv` 和
`outputs/metrics/diffusion_metrics_by_coverage.csv`。旧结果已备份为
`*_white_bias_init.csv` 和 `*_buggy.csv`。

## 如果修了以后还是不够好

现在 denoiser 本身是 ~0.08M 参数的小 UNet、T=100 linear schedule，
能做到的上限就是「接近 pred=mean」（L1_cloud ~ 0.08）。要往 GAN 的
0.022 逼近，考虑：

1. **cosine schedule**：β_max 要小一点或者配合更大 T，让 α̅_T → 0，
   纯噪声初始化就变得 in-distribution。
2. **x0-parameterization**：训练直接回归 target，单步就能出结果，对
   init/schedule 不敏感。
3. **RePaint 风格 resampling**：每步把 mask 外重新用真 target 前向扩散一次，
   往返几次让模型有机会修正。
4. **放大 denoiser**：`base_channels=32` 对 diffusion 太紧，GAN 用同架构
   有 L1-loss 直接监督所以够，diffusion 需要更多容量。
