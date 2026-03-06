#!/usr/bin/env bash
set -euo pipefail

# Use more epochs on the server (GPU if available).
python -m src.train \
  --config configs/default.yaml \
  --output-dir outputs/server \
  --epochs 50

python -m src.evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/server/checkpoints/best.pt \
  --split test
