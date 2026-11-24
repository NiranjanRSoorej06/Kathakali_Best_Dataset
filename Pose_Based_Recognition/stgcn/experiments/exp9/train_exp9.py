from pathlib import Path
import csv
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from dataset_exp9 import KathakaliDataset
from model_exp9 import STGCN


BASE = Path(__file__).resolve().parents[2]
RUN_NAME = "exp9_xy_confidence_single_adjacency_38f"
RUN_DIR = BASE / "runs" / "exp9" / RUN_NAME
LOG_DIR = BASE / "logs" / "exp9" / RUN_NAME

MODEL_PATH = RUN_DIR / "best_model.pt"
LOG_PATH = LOG_DIR / "training_log.csv"
METRICS_PATH = LOG_DIR / "training_metrics.json"

SEED = 42
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 20
NUM_WORKERS = 4


def main():
    for path in (MODEL_PATH, LOG_PATH, METRICS_PATH):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}"
            )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    print("Device:", device)
    print("GPU:", gpu_name)

    train_ds = KathakaliDataset("train")
    val_ds = KathakaliDataset("validation")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )

    adjacency_path = BASE / "shared" / "graph" / "adjacency_single_exp9.npy"
    if not adjacency_path.exists():
        raise FileNotFoundError(adjacency_path)

    adjacency = np.load(adjacency_path)
    model = STGCN(adjacency, num_classes=41, in_channels=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_accuracy",
            "val_loss", "val_accuracy", "val_macro_f1", "learning_rate"
        ])

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training configuration ready.")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses, train_true, train_pred = [], [], []

        for x, labels, mask in train_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_true.extend(labels.cpu().numpy())
            train_pred.extend(outputs.argmax(dim=1).detach().cpu().numpy())

        train_loss = float(np.mean(train_losses))
        train_acc = accuracy_score(train_true, train_pred)

        model.eval()
        val_losses, val_true, val_pred = [], [], []

        with torch.no_grad():
            for x, labels, mask in val_loader:
                x = x.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                outputs = model(x, mask)
                loss = criterion(outputs, labels)

                val_losses.append(loss.item())
                val_true.extend(labels.cpu().numpy())
                val_pred.extend(outputs.argmax(dim=1).cpu().numpy())

        val_loss = float(np.mean(val_losses))
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(
            val_true, val_pred, average="macro", zero_division=0
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}",
            flush=True
        )

        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, train_acc,
                val_loss, val_acc, val_f1, current_lr
            ])

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_accuracy = float(val_acc)
            best_val_f1 = float(val_f1)
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "num_classes": 41,
                "in_channels": 3,
                "seed": SEED
            }, MODEL_PATH)

            print("  -> Best model checkpoint saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    metrics = {
        "experiment": RUN_NAME,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": best_val_accuracy,
        "best_validation_macro_f1": best_val_f1,
        "seed": SEED,
        "epochs_configured": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "device": str(device),
        "gpu_name": gpu_name,
        "checkpoint": str(MODEL_PATH),
        "training_log": str(LOG_PATH)
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print("\nTraining finished.")
    print("Best epoch:", best_epoch)
    print("Checkpoint:", MODEL_PATH)
    print("Metrics:", METRICS_PATH)


if __name__ == "__main__":
    main()
