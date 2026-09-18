#!/usr/bin/env python3
"""Train the canonical full, unpruned SMSE-Net."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from smse_net.dataset import TrainingSegmentDataset
from smse_net.model import CANONICAL_MODEL_NAME, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def learning_rate_at_step(
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    minimum_lr: float,
) -> float:
    if warmup_steps > 0 and global_step < warmup_steps:
        return base_lr * float(global_step + 1) / float(warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((global_step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_lr + (base_lr - minimum_lr) * cosine


def make_loaders(data_dir: Path, batch_size: int, num_workers: int):
    train_set = TrainingSegmentDataset(data_dir, "train")
    validation_set = TrainingSegmentDataset(data_dir, "validation")
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    batch_losses: list[float] = []
    criterion = nn.MSELoss()
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["clean"].to(device, non_blocking=True)
        batch_losses.append(float(criterion(model(inputs), targets).item()))
    if not batch_losses:
        raise RuntimeError("Validation loader is empty.")
    # This is the historical checkpoint criterion: arithmetic mean of batch MSE.
    return float(np.mean(batch_losses))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smse_net.yaml"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/smse_net"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    if config["model"]["name"] != CANONICAL_MODEL_NAME:
        raise ValueError(f"Only {CANONICAL_MODEL_NAME} is supported.")

    seed = int(training["seed"])
    set_seed(seed)
    device = torch.device(args.device)
    model = build_model().to(device)
    train_loader, validation_loader = make_loaders(
        args.data_dir,
        int(training["batch_size"]),
        int(training["num_workers"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.MSELoss()
    epochs = int(training["epochs"])
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * float(training["warmup_fraction"]))
    base_lr = float(training["learning_rate"])
    minimum_lr = float(training["minimum_learning_rate"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    best_validation_mse = float("inf")
    global_step = 0

    print(f"Model: {CANONICAL_MODEL_NAME}")
    print(f"Training segments: {len(train_loader.dataset)}")
    print(f"Validation segments: {len(validation_loader.dataset)}")
    print(f"Total optimization steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            learning_rate = learning_rate_at_step(
                global_step, total_steps, warmup_steps, base_lr, minimum_lr
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["clean"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            global_step += 1

        validation_mse = validate(model, validation_loader, device)
        train_mse = float(np.mean(train_losses))
        print(
            f"Epoch {epoch:03d}/{epochs}: train_mse={train_mse:.12f}, "
            f"validation_mse={validation_mse:.12f}, lr={optimizer.param_groups[0]['lr']:.10g}"
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_validation_mse,
                    "args": {
                        "model_name": CANONICAL_MODEL_NAME,
                        "sample_rate": int(config["audio"]["sample_rate"]),
                        "n_freq": int(config["model"]["frequency_bins"]),
                    },
                    "normalization_stats": {
                        "acc_min": -100.0,
                        "acc_max": 60.0,
                        "acoustic_min": -100.0,
                        "acoustic_max": 60.0,
                        "normalization_type": "fixed_db_ref1",
                    },
                },
                args.output_dir / "best_model.pth",
            )
            print(f"Saved new best checkpoint at epoch {epoch}.")


if __name__ == "__main__":
    main()
