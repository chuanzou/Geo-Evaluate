#!/usr/bin/env bash
# Re-run the diffusion evaluation ONLY (don't retrain; don't touch GAN).
# Use after fixing inpaint() init — will overwrite
# outputs/metrics/diffusion_metrics{,_by_coverage}.csv.
#
# Activate your venv first (so torch/mps/etc. are available), then:
#     bash scripts/rerun_diffusion_eval.sh

set -euo pipefail
cd "$(dirname "$0")/.."

python -c "
import yaml, torch
from cloud_removal.training.evaluate import evaluate_method

with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)
device = torch.device(config.get('device', 'cpu'))
evaluate_method(config, 'diffusion', device)
"
