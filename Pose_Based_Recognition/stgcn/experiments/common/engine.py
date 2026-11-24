import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]

SEED = 42
EPOCHS = 100
BATCH_SIZE = 16
LR = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 20

def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def paths(exp):
    names = {
        4: "exp4_xy_single_adjacency",
        5: "exp5_xy_confidence_3partition",
        6: "exp6_xy_3partition",
    }
    name = names[exp]
    run = ROOT / "runs" / f"exp{exp}" / name
    log = ROOT / "logs" / f"exp{exp}" / name
    return run, log

def make_model(build_model, device):
    return build_model().to(device)

def forward(model, x, mask, use_mask):
    if use_mask:
        return model(x, mask)
    return model(x.unsqueeze(-1))

def evaluate_loader(model, loader, criterion, device, use_mask):
    model.eval()
    losses, targets, predictions = [], [], []

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            logits = forward(model, x, mask, use_mask)
            loss = criterion(logits, y)

            losses.append(loss.item())
            targets.extend(y.cpu().numpy())
            predictions.extend(logits.argmax(1).cpu().numpy())

    return (
        float(np.mean(losses)),
        accuracy_score(targets, predictions),
        f1_score(targets, predictions, average="macro", zero_division=0),
        np.asarray(targets),
        np.asarray(predictions)
    )

def train(exp, Dataset, build_model, use_mask):
    seed_everything()
    run_dir, log_dir = paths(exp)
    checkpoint_path = run_dir / "best_model.pt"
    history_path = log_dir / "training_log.csv"
    metrics_path = log_dir / "best_validation_metrics.json"

    resume_checkpoint = None
    if exp == 5 and checkpoint_path.exists() and history_path.exists():
        resume_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        print("Resuming Exp5 from checkpoint epoch:", resume_checkpoint["epoch"])
    elif checkpoint_path.exists() or history_path.exists() or metrics_path.exists():
        raise FileExistsError(
            f"Exp{exp} output already exists. Refusing to overwrite."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Experiment:", exp)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = Dataset("train")
    val_ds = Dataset("val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    model = make_model(build_model, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    print("Train:", len(train_ds), "Validation:", len(val_ds))
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    history = []
    start_epoch = 1

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        best_loss = resume_checkpoint["best_val_loss"]
        best_epoch = resume_checkpoint["epoch"]
        history = pd.read_csv(history_path).to_dict("records")
        history = [r for r in history if int(r["epoch"]) <= best_epoch]
        pd.DataFrame(history).to_csv(history_path, index=False)
        start_epoch = best_epoch + 1
        print("Restored model and optimizer.")
        print("Best validation loss:", best_loss)
        print("Continuing from epoch:", start_epoch)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_losses, targets, predictions = [], [], []

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = forward(model, x, mask, use_mask)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(loss.item())
            targets.extend(y.cpu().numpy())
            predictions.extend(logits.argmax(1).detach().cpu().numpy())

        train_loss = float(np.mean(train_losses))
        train_acc = accuracy_score(targets, predictions)

        val_loss, val_acc, val_f1, _, _ = evaluate_loader(
            model, val_loader, criterion, device, use_mask
        )
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "learning_rate": current_lr
        }
        history.append(record)
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f} | lr={current_lr:.6g}",
            flush=True
        )

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            wait = 0

            torch.save({
                "experiment": exp,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "seed": SEED,
                "class_names": sorted(
                    pd.read_csv(ROOT / "datasets/processed/metadata_split.csv")
                    [["label_index", "class_name"]]
                    .drop_duplicates()
                    .sort_values("label_index")["class_name"].tolist()
                )
            }, checkpoint_path)
            print("  Best checkpoint saved.")
        else:
            wait += 1

        if wait >= PATIENCE:
            print("Early stopping.")
            break

    metrics_path.write_text(json.dumps({
        "experiment": exp,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "seed": SEED,
        "epochs_configured": EPOCHS,
        "checkpoint": str(checkpoint_path),
        "training_log": str(history_path)
    }, indent=2) + "\n")

    print("\nTraining complete.")
    print("Best epoch:", best_epoch)
    print("Checkpoint:", checkpoint_path)
    print("Log:", history_path)

def test(exp, Dataset, build_model, use_mask):
    seed_everything()
    run_dir, log_dir = paths(exp)
    checkpoint_path = run_dir / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    outputs = [
        log_dir / "test_metrics.json",
        log_dir / "classification_report.csv",
        log_dir / "confusion_matrix.csv",
        log_dir / "predictions.csv"
    ]
    existing = [p for p in outputs if p.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    log_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Dataset("test")
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    model = make_model(build_model, device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    criterion = nn.CrossEntropyLoss()
    loss, acc, f1, y_true, y_pred = evaluate_loader(
        model, loader, criterion, device, use_mask
    )

    metadata = pd.read_csv(ROOT / "datasets/processed/metadata_split.csv")
    class_names = (
        metadata[["label_index", "class_name"]]
        .drop_duplicates()
        .sort_values("label_index")["class_name"].tolist()
    )
    labels = list(range(len(class_names)))

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    pd.DataFrame(report).transpose().to_csv(
        log_dir / "classification_report.csv"
    )
    pd.DataFrame(
        cm, index=class_names, columns=class_names
    ).to_csv(log_dir / "confusion_matrix.csv")

    pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": y_true == y_pred
    }).to_csv(log_dir / "predictions.csv", index=False)

    result = {
        "experiment": exp,
        "checkpoint_epoch": checkpoint["epoch"],
        "test_samples": len(dataset),
        "test_loss": loss,
        "test_accuracy": acc,
        "test_macro_f1": f1
    }
    (log_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print("\nTEST RESULTS")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("Reports saved to:", log_dir)
