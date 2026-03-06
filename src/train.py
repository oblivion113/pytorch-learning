from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from .data import create_data_bundle
from .model import MLPClassifier
from .utils import ensure_dir, get_device, load_config, save_json, set_seed


def _run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        if is_train:
            loss.backward()
            optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        total_correct += (predictions == batch_y).sum().item()
        total_loss += loss.item() * batch_y.size(0)
        total_samples += batch_y.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def run_training(config: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    device = get_device()

    experiment_cfg = config["experiment"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    output_root = Path(output_dir or experiment_cfg["output_dir"])
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    ensure_dir(output_root / "metrics")

    data_bundle = create_data_bundle(config)
    model = MLPClassifier(
        input_dim=data_bundle.input_dim,
        num_classes=data_bundle.num_classes,
        hidden_dims=list(model_cfg["hidden_dims"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(train_cfg["epochs"])
    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_checkpoint_path = checkpoints_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            dataloader=data_bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = _run_epoch(
            model=model,
            dataloader=data_bundle.val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )
        test_metrics = _run_epoch(
            model=model,
            dataloader=data_bundle.test_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        epoch_log = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
        history.append(epoch_log)

        tqdm.write(
            "epoch=%d train_acc=%.4f val_acc=%.4f test_acc=%.4f"
            % (
                epoch,
                train_metrics["accuracy"],
                val_metrics["accuracy"],
                test_metrics["accuracy"],
            )
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": data_bundle.input_dim,
                    "num_classes": data_bundle.num_classes,
                    "model_config": model_cfg,
                    "best_val_accuracy": best_val_accuracy,
                },
                best_checkpoint_path,
            )

    summary = {
        "device": str(device),
        "epochs": epochs,
        "best_val_accuracy": best_val_accuracy,
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "final_test_accuracy": history[-1]["test_accuracy"],
        "checkpoint_path": str(best_checkpoint_path),
    }
    save_json({"summary": summary, "history": history}, output_root / "metrics" / "train_metrics.json")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a simple PyTorch classifier.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    summary = run_training(config=config, output_dir=args.output_dir)
    print(summary)


if __name__ == "__main__":
    main()
