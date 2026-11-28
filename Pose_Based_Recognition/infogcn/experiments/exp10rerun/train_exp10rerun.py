"""Training script for EXP10RERUN — Isolated InfoGCN Baseline rerun on the 430-video 300/65/65 split."""

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

INFOGCN_ROOT = Path(__file__).resolve().parents[2]
STGCN_ROOT = INFOGCN_ROOT.parent / "stgcn"
if str(INFOGCN_ROOT) not in sys.path:
    sys.path.insert(0, str(INFOGCN_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from experiments.exp10rerun.model_exp10rerun import InfoGCN, InfoGCNLoss

METADATA_PATH = STGCN_ROOT / "datasets/processed/metadata_split.csv"
DEFAULT_RUN_NAME = "exp10rerun_infogcn_baseline"


def _load_kathakali_dataset():
    """Load read-only KathakaliDataset from stgcn without writing bytecode there."""
    dataset_path = STGCN_ROOT / "experiments/common/dataset.py"
    prev_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("stgcn_common_dataset", dataset_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load dataset module from {dataset_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.KathakaliDataset
    finally:
        sys.dont_write_bytecode = prev_dont_write


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader worker RNGs deterministically from PyTorch's per-worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_class_names() -> list[str]:
    metadata = pd.read_csv(METADATA_PATH)
    return (
        metadata[["label_index", "class_name"]]
        .drop_duplicates()
        .sort_values("label_index")["class_name"]
        .tolist()
    )


def validate_run_name(run_name: str) -> str:
    """Prevent path traversal so outputs remain strictly inside runs/exp10rerun and logs/exp10rerun."""
    if not run_name or run_name in {".", ".."} or "/" in run_name or "\\" in run_name:
        raise ValueError(f"Invalid --run-name '{run_name}': must be a single directory name.")
    return run_name


def evaluate_loader(
    model: InfoGCN,
    loader: DataLoader,
    criterion: InfoGCNLoss,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    targets, predictions = [], []

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits, mu, logvar, z = model(x, mask, return_latent=True)
            loss = criterion(logits, y, mu=mu, logvar=logvar, z=z, kl_scale=1.0)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            targets.extend(y.cpu().numpy())
            predictions.extend(logits.argmax(dim=1).cpu().numpy())

    val_loss = float(total_loss / total_samples)
    val_acc = float(accuracy_score(targets, predictions))
    val_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
    return val_loss, val_acc, val_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EXP10RERUN InfoGCN baseline on Kathakali")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--selection-metric", choices=["val_loss", "val_accuracy"], default="val_loss")
    parser.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--use-ib", action="store_true", help="Enable VIB + MMD latent bottleneck")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--lambda-kl", type=float, default=1e-4)
    parser.add_argument("--lambda-mmd", type=float, default=1e-3)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = validate_run_name(args.run_name)
    seed_everything(args.seed)

    run_dir = INFOGCN_ROOT / "runs/exp10rerun" / run_name
    log_dir = INFOGCN_ROOT / "logs/exp10rerun" / run_name
    checkpoint_path = run_dir / "best_model.pt"
    history_path = log_dir / "training_log.csv"
    metrics_path = log_dir / "best_validation_metrics.json"

    for path in (checkpoint_path, history_path, metrics_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing EXP10RERUN output: {path}")

    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    pin_memory = device.type == "cuda"

    class_names = load_class_names()
    num_classes = len(class_names)

    KathakaliDataset = _load_kathakali_dataset()
    train_ds = KathakaliDataset("train", in_channels=3)
    val_ds = KathakaliDataset("val", in_channels=3)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model_config = {
        "num_classes": num_classes,
        "in_channels": 3,
        "num_joints": 50,
        "base_channels": args.base_channels,
        "dropout": args.dropout,
        "head_dropout": args.head_dropout,
        "use_ib": args.use_ib,
        "latent_dim": args.latent_dim,
    }
    model = InfoGCN(**model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = InfoGCNLoss(
        label_smoothing=args.label_smoothing,
        use_ib=args.use_ib,
        lambda_kl=args.lambda_kl,
        lambda_mmd=args.lambda_mmd,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=6
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )

    print("=" * 65)
    print(f"EXP10RERUN — InfoGCN Baseline ({run_name})")
    print("=" * 65)
    print("Device:", device, f"({gpu_name})" if gpu_name else "")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")
    print(f"Max epochs: {args.epochs} | Patience: {args.patience} | Selection metric: {args.selection_metric}")
    print(f"Use IB: {args.use_ib} | Checkpoint dir: {run_dir} | Log dir: {log_dir}")

    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "learning_rate",
        ])

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_val_f1 = 0.0
    best_epoch = 0
    wait = 0
    stopped_early = False
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0
        targets, predictions = [], []
        kl_scale = min(1.0, epoch / max(1, args.kl_warmup_epochs)) if args.use_ib else 1.0

        for x, y, mask in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar, z = model(x, mask, return_latent=True)
            loss = criterion(logits, y, mu=mu, logvar=logvar, z=z, kl_scale=kl_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = y.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size
            targets.extend(y.cpu().numpy())
            predictions.extend(logits.argmax(dim=1).detach().cpu().numpy())

        train_loss = float(total_train_loss / total_train_samples)
        train_acc = float(accuracy_score(targets, predictions))

        val_loss, val_acc, val_f1 = evaluate_loader(model, val_loader, criterion, device)

        if args.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[EXP10RERUN] Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
            f"lr={current_lr:.6g}",
            flush=True,
        )

        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_f1,
                current_lr,
            ])

        if args.selection_metric == "val_loss":
            improved = val_loss < best_val_loss - 1e-4
        else:
            improved = val_acc > best_val_acc + 1e-4

        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            wait = 0

            torch.save(
                {
                    "experiment": "exp10rerun",
                    "run_name": run_name,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model_config,
                    "best_val_loss": best_val_loss,
                    "val_accuracy": best_val_acc,
                    "val_macro_f1": best_val_f1,
                    "seed": args.seed,
                    "class_names": class_names,
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                },
                checkpoint_path,
            )
            print(
                f"  -> [EXP10RERUN] Best checkpoint saved at epoch {epoch:03d} "
                f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}, val_f1={best_val_f1:.4f}).",
                flush=True,
            )
        else:
            wait += 1
            print(
                f"  -> [EXP10RERUN] No improvement ({wait}/{args.patience} without improvement).",
                flush=True,
            )

        if wait >= args.patience:
            stopped_early = True
            print(f"[EXP10RERUN] Early stopping triggered at epoch {epoch:03d}/{args.epochs} (patience={args.patience}).", flush=True)
            break

    metrics_payload = {
        "experiment": "exp10rerun",
        "run_name": run_name,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "stopped_early": stopped_early,
        "selection_metric": args.selection_metric,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": best_val_acc,
        "best_validation_macro_f1": best_val_f1,
        "seed": args.seed,
        "epochs_configured": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "model_config": model_config,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device),
        "gpu_name": gpu_name,
        "checkpoint": str(checkpoint_path),
        "training_log": str(history_path),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 65)
    print("EXP10RERUN Training Summary")
    print("=" * 65)
    print(f"Completion status    : {'Early stopped' if stopped_early else 'Completed max epochs'} at epoch {last_epoch}/{args.epochs}")
    print(f"Best epoch           : {best_epoch}")
    print(f"Best val_loss        : {best_val_loss:.4f}")
    print(f"Best val_accuracy    : {best_val_acc:.4f}")
    print(f"Best val_macro_f1    : {best_val_f1:.4f}")
    print(f"Checkpoint saved to  : {checkpoint_path}")
    print(f"Training log saved   : {history_path}")
    print(f"Val metrics saved    : {metrics_path}")


if __name__ == "__main__":
    main()
