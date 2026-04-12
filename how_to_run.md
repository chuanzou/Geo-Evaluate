# How to Run the Full Cloud Removal Benchmark

This document explains how to run the full project on a new computer, including a GPU machine.

The project follows this pipeline:

```text
Google Earth Engine Sentinel-2 export
-> download GeoTIFF files from Google Drive
-> convert GeoTIFF files to .npy ground-truth patches
-> generate synthetic cloud masks and cloudy images
-> run temporal baseline
-> train/evaluate GAN
-> train/evaluate diffusion inpainting
-> compare metrics and curves
```

## 1. Clone or Copy the Repository

Move into the project folder:

```bash
cd /path/to/Geo-Evaluate
```

If you clone it from GitHub later:

```bash
git clone <your-repo-url>
cd Geo-Evaluate
```

## 2. Create a Python Environment

Recommended Python version:

```text
Python 3.10 or newer
```

Using conda:

```bash
conda create -n geo-evaluate python=3.10 -y
conda activate geo-evaluate
```

Or using venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

Install base dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the project package in editable mode, including geospatial/GEE dependencies:

```bash
python -m pip install -e ".[geo]"
```

If the GPU machine needs a specific PyTorch CUDA build, install PyTorch from the official PyTorch command first, then run the two commands above.

Check whether PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 4. Authenticate Google Earth Engine

Only needed on the machine that will create Earth Engine export tasks.

```bash
earthengine authenticate
```

Follow the browser login flow.

## 5. Export Sentinel-2 Beijing Patches from Google Earth Engine

The exporter uses:

```text
Dataset: COPERNICUS/S2_SR_HARMONIZED
Area: Beijing and surrounding areas
AOI: [115.4, 39.4, 117.6, 41.1]
Bands: red, green, blue, nir
Original Sentinel-2 bands: B4, B3, B2, B8
Default date range: April 1 to October 31
Default year: 2023
Default patch size: 128 x 128 pixels
Default scale: 10 meters
```

First run a small test export:

```bash
python scripts/download_sentinel2_beijing.py \
  --year 2023 \
  --num-patches 20 \
  --max-tasks 20 \
  --drive-folder GeoEvaluate_Beijing_S2_Test
```

After confirming that files arrive in Google Drive, run the full export:

```bash
python scripts/download_sentinel2_beijing.py \
  --year 2023 \
  --num-patches 2000 \
  --max-tasks 2000 \
  --drive-folder GeoEvaluate_Beijing_S2
```

The exported files will appear in your Google Drive folder:

```text
GeoEvaluate_Beijing_S2/
```

Earth Engine exports can take time. Check task status in the Earth Engine Tasks page or with the Earth Engine CLI.

## 6. Download GeoTIFF Files Locally

After GEE finishes exporting, download the GeoTIFF files from Google Drive to:

```text
data/raw/sentinel2_beijing/
```

Create the directory if needed:

```bash
mkdir -p data/raw/sentinel2_beijing
```

Expected raw data layout:

```text
data/raw/sentinel2_beijing/
  s2_beijing_patch_2023_00000.tif
  s2_beijing_patch_2023_00001.tif
  ...
```

## 7. Convert GeoTIFF Files to .npy Ground Truth Patches

The training pipeline expects cloud-free ground-truth patches in this format:

```text
Shape: (4, H, W)
Channel order: red, green, blue, nir
Value range: [0, 1]
File type: .npy
```

Expected output directory:

```text
data/processed/synthetic_cloud_benchmark/ground_truth/
```

The intended command is:

```bash
python scripts/convert_geotiff_to_npy.py \
  --input-dir data/raw/sentinel2_beijing \
  --output-dir data/processed/synthetic_cloud_benchmark/ground_truth
```

Important: this conversion script still needs to be added before running the full real-data pipeline.

## 8. Generate Synthetic Cloud Benchmark

After `.npy` ground-truth patches exist, generate synthetic clouds:

```bash
python scripts/prepare_synthetic_dataset.py --from-ground-truth
```

This creates:

```text
data/processed/synthetic_cloud_benchmark/
  ground_truth/
  cloudy/
  masks/
  splits/
    train.txt
    val.txt
    test.txt
```

The default cloud coverage levels are:

```text
5%, 10%, 30%, 50%, 70%
```

## 9. Run the Multi-Temporal Baseline

The temporal baseline has no trainable parameters.

```bash
python scripts/run_experiment.py \
  --method temporal \
  --config configs/default.yaml
```

Expected outputs:

```text
outputs/metrics/temporal_metrics.csv
outputs/figures/temporal_curves.png
```

## 10. Train and Evaluate the GAN Model

```bash
python scripts/run_experiment.py \
  --method gan \
  --config configs/default.yaml \
  --train
```

Expected outputs:

```text
outputs/checkpoints/gan_generator.pt
outputs/checkpoints/gan_generator_best.pt
outputs/metrics/gan_metrics.csv
outputs/figures/gan_curves.png
```

`gan_generator_best.pt` is selected by the lowest validation cloud-region L1 loss. Later evaluation code loads this best checkpoint automatically when it exists.

## 11. Train and Evaluate the Diffusion Model

```bash
python scripts/run_experiment.py \
  --method diffusion \
  --config configs/default.yaml \
  --train
```

Expected outputs:

```text
outputs/checkpoints/diffusion.pt
outputs/checkpoints/diffusion_best.pt
outputs/metrics/diffusion_metrics.csv
outputs/figures/diffusion_curves.png
```

`diffusion_best.pt` is selected by the lowest validation noise MSE. Later evaluation code loads this best checkpoint automatically when it exists.

## 12. Generate Qualitative Interpretation Figures

After training the best GAN and diffusion checkpoints, generate figures for qualitative analysis:

```bash
python scripts/interpretation.py \
  --config configs/default.yaml \
  --split test \
  --num-samples 3
```

Expected outputs:

```text
outputs/figures/interpretation/cloud_coverage_examples.png
outputs/figures/interpretation/model_completions_*.png
```

These figures show:

```text
different cloud coverage examples
cloud masks
ground truth images
multi-temporal baseline output
GAN best-model output
diffusion best-model output
```

`interpretation.py` automatically loads:

```text
outputs/checkpoints/gan_generator_best.pt
outputs/checkpoints/diffusion_best.pt
```

If a best checkpoint does not exist, the script falls back to the final checkpoint. If neither exists, that model is skipped.

## 13. Run Smoke Tests

Run this after setup or after changing code:

```bash
pytest -q
```

Expected result:

```text
3 passed
```

## 14. Main Files to Inspect

Configuration:

```text
configs/default.yaml
```

GEE export:

```text
scripts/download_sentinel2_beijing.py
```

Synthetic cloud generation:

```text
scripts/prepare_synthetic_dataset.py
src/cloud_removal/data/synthetic_clouds.py
```

Models:

```text
src/cloud_removal/models/baseline.py
src/cloud_removal/models/unet.py
src/cloud_removal/models/diffusion.py
```

Training and evaluation:

```text
src/cloud_removal/training/train_gan.py
src/cloud_removal/training/train_diffusion.py
src/cloud_removal/training/evaluate.py
scripts/interpretation.py
```

Metrics:

```text
src/cloud_removal/evaluation/metrics.py
src/cloud_removal/evaluation/plots.py
```

## 15. Recommended Full Command Order

For a real full run, the intended command order is:

```bash
cd /path/to/Geo-Evaluate

conda create -n geo-evaluate python=3.10 -y
conda activate geo-evaluate

python -m pip install -r requirements.txt
python -m pip install -e ".[geo]"

earthengine authenticate

python scripts/download_sentinel2_beijing.py \
  --year 2023 \
  --num-patches 2000 \
  --max-tasks 2000 \
  --drive-folder GeoEvaluate_Beijing_S2

mkdir -p data/raw/sentinel2_beijing

# Download GeoTIFF files from Google Drive into data/raw/sentinel2_beijing/

python scripts/convert_geotiff_to_npy.py \
  --input-dir data/raw/sentinel2_beijing \
  --output-dir data/processed/synthetic_cloud_benchmark/ground_truth

python scripts/prepare_synthetic_dataset.py --from-ground-truth

python scripts/run_experiment.py --method temporal --config configs/default.yaml
python scripts/run_experiment.py --method gan --config configs/default.yaml --train
python scripts/run_experiment.py --method diffusion --config configs/default.yaml --train

python scripts/interpretation.py --config configs/default.yaml --split test --num-samples 3
```

## 16. Notes for GPU Runs

The project uses `device: auto` in:

```text
configs/default.yaml
```

This means:

```text
Use CUDA if available, otherwise use CPU.
```

For larger GPU runs, consider increasing:

```yaml
training:
  batch_size: 8
  epochs: 5
```

For formal experiments, run more than 5 epochs. The current default is intentionally small so the pipeline can be tested quickly.

## 17. Current Limitation

The project already supports:

```text
GEE export task creation
synthetic cloud generation
temporal baseline
GAN training/evaluation
diffusion training/evaluation
best checkpoint saving
metrics and plots
```

The missing bridge is:

```text
GeoTIFF -> .npy conversion
```

Add `scripts/convert_geotiff_to_npy.py` before running the full real-data experiment end to end.
