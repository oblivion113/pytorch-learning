from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    num_classes: int


def create_data_bundle(config: dict[str, Any]) -> DataBundle:
    data_cfg = config["data"]
    seed = int(config["seed"])

    features, labels = make_moons(
        n_samples=int(data_cfg["num_samples"]),
        noise=float(data_cfg["noise"]),
        random_state=seed,
    )

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        features,
        labels,
        test_size=float(data_cfg["test_size"]),
        random_state=seed,
        stratify=labels,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=float(data_cfg["val_size"]),
        random_state=seed,
        stratify=y_trainval,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    def to_dataset(x, y) -> TensorDataset:
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return TensorDataset(x_tensor, y_tensor)

    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        to_dataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        to_dataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        to_dataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_dim=x_train.shape[1],
        num_classes=len(set(labels.tolist())),
    )
