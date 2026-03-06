#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
  --config configs/default.yaml \
  --output-dir outputs/local \
  --epochs 10

python -m src.evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/local/checkpoints/best.pt \
  --split test
