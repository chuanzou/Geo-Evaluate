# Geo-Evaluate: Cloud Removal Benchmark

This project implements the code framework for the proposal:
**Evaluating Generative Models vs. Multi-Temporal Fusion for Cloud Removal in Satellite Imagery**.

The benchmark follows the proposal design:

- Ground truth: cloud-free Sentinel-2 style RGB+NIR image patches.
- Synthetic clouds: controlled cloud cover ratios of 5%, 10%, 30%, 50%, and 70%.
- Methods:
  - Multi-temporal fusion baseline.
  - GAN-based image-to-image model.
  - Diffusion-style mask-guided inpainting model.
- Metrics:
  - PSNR.
  - SSIM.
  - NDVI MAE.

## Folder Plan

```text
configs/                 Experiment configuration.
data/raw/                Original Sentinel-2 exports or downloaded imagery.
data/processed/          Patch datasets used by training/evaluation.
data/splits/             Train/test split files.
notebooks/               Exploratory analysis and visual inspection.
outputs/checkpoints/     Model checkpoints.
outputs/figures/         Metric curves and visual comparisons.
outputs/metrics/         Evaluation CSV files.
scripts/                 Command-line entry points.
src/cloud_removal/       Reusable research code.
tests/                   Smoke tests for data, metrics, and models.
```

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Create a small synthetic benchmark to test the pipeline:

```bash
python scripts/prepare_synthetic_dataset.py --num-images 64 --image-size 128
```

Evaluate the multi-temporal baseline:

```bash
python scripts/run_experiment.py --method temporal --config configs/default.yaml
```

Train and evaluate a GAN model:

```bash
python scripts/run_experiment.py --method gan --config configs/default.yaml --train
```

GAN training saves both the final checkpoint and the validation-best checkpoint:

```text
outputs/checkpoints/gan_generator.pt
outputs/checkpoints/gan_generator_best.pt
```

Train and evaluate the diffusion-style inpainting model:

```bash
python scripts/run_experiment.py --method diffusion --config configs/default.yaml --train
```

Diffusion training saves both the final checkpoint and the validation-best checkpoint:

```text
outputs/checkpoints/diffusion.pt
outputs/checkpoints/diffusion_best.pt
```

Generate qualitative interpretation figures after training:

```bash
python scripts/interpretation.py --config configs/default.yaml --split test --num-samples 3
```

Figures are written to:

```text
outputs/figures/interpretation/
```

## Expected Real Data Format

For real Sentinel-2 patches, save each cloud-free patch as `.npy` with shape:

```text
(4, H, W)
```

The channel order should be:

```text
red, green, blue, nir
```

Values should be normalized to `[0, 1]`. Place files under:

```text
data/processed/synthetic_cloud_benchmark/ground_truth/
```

Then run `scripts/prepare_synthetic_dataset.py --from-ground-truth` to create synthetic cloudy inputs and masks.

## Google Earth Engine Export

Use the Python exporter to start Sentinel-2 Beijing patch exports from Earth Engine:

```bash
python -m pip install ".[geo]"
earthengine authenticate
python scripts/download_sentinel2_beijing.py --year 2023 --num-patches 2000 --drive-folder GeoEvaluate_Beijing_S2
```
