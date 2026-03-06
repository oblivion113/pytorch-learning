from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import DataBundle, create_data_bundle
from .model import MLPClassifier
from .train import _run_epoch
from .utils import get_device, load_config, save_json, set_seed


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict[str, float]:
    set_seed(int(config["seed"]))
    device = get_device()

    data_bundle: DataBundle = create_data_bundle(config)
    split_to_loader = {
        "train": data_bundle.train_loader,
        "val": data_bundle.val_loader,
        "test": data_bundle.test_loader,
    }
    if split not in split_to_loader:
        raise ValueError(f"Unsupported split '{split}'. Use one of train/val/test.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint["model_config"]

    model = MLPClassifier(
        input_dim=int(checkpoint["input_dim"]),
        num_classes=int(checkpoint["num_classes"]),
        hidden_dims=list(model_cfg["hidden_dims"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    metrics = _run_epoch(
        model=model,
        dataloader=split_to_loader[split],
        criterion=criterion,
        optimizer=None,
        device=device,
    )
    return {"loss": metrics["loss"], "accuracy": metrics["accuracy"], "split": split}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save-json", type=str, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(config["experiment"]["output_dir"]) / "checkpoints" / "best.pt")
    metrics = evaluate_checkpoint(config=config, checkpoint_path=checkpoint, split=args.split)
    print(metrics)

    if args.save_json:
        save_json(metrics, args.save_json)


if __name__ == "__main__":
    main()
