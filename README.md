# PyTorch Learning Project

This is a beginner-friendly PyTorch project with a full train/evaluate workflow.
It uses a synthetic classification dataset (`make_moons`) so you can learn model training without downloading data.

## Project structure

- `configs/default.yaml`: all main hyperparameters.
- `src/data.py`: dataset creation, split, and DataLoader setup.
- `src/model.py`: a small MLP classifier.
- `src/train.py`: training loop + checkpoint saving.
- `src/evaluate.py`: checkpoint evaluation on train/val/test splits.
- `tests/test_smoke.py`: quick smoke tests.
- `scripts/run_local.sh`: local CPU run.
- `scripts/run_server.sh`: longer server run.

## Setup

Python version: `>=3.10` (also defined in `pyproject.toml`).

CPU/local install:

```bash
conda activate <your_env_name>
python -m pip install -r requirements.txt
```

GPU/server install (CUDA 12.4 wheels):

```bash
conda activate <your_env_name>
python -m pip install -r requirements-gpu.txt
```

## Run locally

```bash
bash scripts/run_local.sh
```

Or run commands directly:

```bash
python -m src.train --config configs/default.yaml --output-dir outputs/local --epochs 10
python -m src.evaluate --config configs/default.yaml --checkpoint outputs/local/checkpoints/best.pt --split test
```

## Run tests

```bash
pytest -q
```

## Learning path

1. Change one setting in `configs/default.yaml` (for example, `train.learning_rate`).
2. Run training again and compare final test accuracy.
3. Modify `src/model.py` hidden layer sizes.
4. Add one more metric (for example precision/recall) in `src/evaluate.py`.
