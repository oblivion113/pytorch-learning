from __future__ import annotations

from pathlib import Path

import torch

from src.data import create_data_bundle
from src.evaluate import evaluate_checkpoint
from src.model import MLPClassifier
from src.train import run_training


def _tiny_config() -> dict:
    return {
        "seed": 42,
        "experiment": {"name": "test_run", "output_dir": "outputs/test_run"},
        "data": {
            "num_samples": 400,
            "noise": 0.2,
            "test_size": 0.2,
            "val_size": 0.2,
            "batch_size": 32,
            "num_workers": 0,
        },
        "model": {"hidden_dims": [16, 16], "dropout": 0.0},
        "train": {"epochs": 2, "learning_rate": 0.01, "weight_decay": 0.0},
    }


def test_data_bundle_shapes() -> None:
    data_bundle = create_data_bundle(_tiny_config())
    batch_x, batch_y = next(iter(data_bundle.train_loader))
    assert batch_x.ndim == 2
    assert batch_y.ndim == 1
    assert batch_x.shape[1] == data_bundle.input_dim == 2
    assert data_bundle.num_classes == 2


def test_model_forward_shape() -> None:
    model = MLPClassifier(input_dim=2, num_classes=2, hidden_dims=[8, 8], dropout=0.0)
    logits = model(torch.randn(4, 2))
    assert logits.shape == (4, 2)


def test_train_and_eval_smoke(tmp_path: Path) -> None:
    config = _tiny_config()
    output_dir = tmp_path / "train_outputs"
    summary = run_training(config=config, output_dir=output_dir)
    checkpoint_path = output_dir / "checkpoints" / "best.pt"

    assert checkpoint_path.exists()
    assert 0.0 <= summary["best_val_accuracy"] <= 1.0

    metrics = evaluate_checkpoint(config=config, checkpoint_path=checkpoint_path, split="test")
    assert 0.0 <= metrics["accuracy"] <= 1.0
