"""Evaluation script for EXP10 — InfoGCN Baseline on the 65-video Kathakali test set."""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

INFOGCN_ROOT = Path(__file__).resolve().parents[2]
STGCN_ROOT = INFOGCN_ROOT.parent / "stgcn"
if str(INFOGCN_ROOT) not in sys.path:
    sys.path.insert(0, str(INFOGCN_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from experiments.exp10.model_exp10 import InfoGCN, InfoGCNLoss

METADATA_PATH = STGCN_ROOT / "datasets/processed/metadata_split.csv"
DEFAULT_RUN_NAME = "exp10_infogcn_baseline"


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


def load_class_names() -> list[str]:
    metadata = pd.read_csv(METADATA_PATH)
    return (
        metadata[["label_index", "class_name"]]
        .drop_duplicates()
        .sort_values("label_index")["class_name"]
        .tolist()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EXP10 InfoGCN baseline on test split")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = INFOGCN_ROOT / "runs/exp10" / args.run_name
    log_dir = INFOGCN_ROOT / "logs/exp10" / args.run_name
    checkpoint_path = args.checkpoint if args.checkpoint is not None else (run_dir / "best_model.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_files = [
        log_dir / "test_metrics.json",
        log_dir / "classification_report.csv",
        log_dir / "test_classification_report.txt",
        log_dir / "confusion_matrix.csv",
        log_dir / "predictions.csv",
    ]
    existing = [str(p) for p in output_files if p.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing test outputs:\n" + "\n".join(existing))

    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    print("Evaluation device:", device, f"({gpu_name})" if gpu_name else "")

    class_names = load_class_names()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_classes = checkpoint.get("class_names")
    if ckpt_classes is not None and list(ckpt_classes) != class_names:
        raise ValueError("Checkpoint class_names do not match stgcn/datasets/processed/metadata_split.csv")

    model_config = checkpoint.get(
        "model_config",
        {
            "num_classes": len(class_names),
            "in_channels": 3,
            "num_joints": 50,
            "base_channels": 48,
            "dropout": 0.15,
            "head_dropout": 0.3,
            "use_ib": False,
            "latent_dim": 128,
        },
    )

    model = InfoGCN(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    KathakaliDataset = _load_kathakali_dataset()
    test_ds = KathakaliDataset("test", in_channels=3)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    criterion = InfoGCNLoss(use_ib=bool(model_config.get("use_ib", False)))
    total_loss = 0.0
    total_samples = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits, mu, logvar, z = model(x, mask, return_latent=True)
            loss = criterion(logits, y, mu=mu, logvar=logvar, z=z)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    test_loss = float(total_loss / total_samples)
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_precision = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_txt = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    pd.DataFrame(report_dict).transpose().to_csv(log_dir / "classification_report.csv")
    (log_dir / "test_classification_report.txt").write_text(report_txt, encoding="utf-8")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(log_dir / "confusion_matrix.csv")

    with open(log_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_id", "true_class", "predicted_id", "predicted_class", "correct"])
        for t, p in zip(y_true, y_pred):
            writer.writerow([t, class_names[t], p, class_names[p], t == p])

    metrics = {
        "experiment": 10,
        "run_name": args.run_name,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_samples": len(test_ds),
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "test_macro_precision": macro_precision,
        "test_macro_recall": macro_recall,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device),
        "gpu_name": gpu_name,
        "checkpoint": str(checkpoint_path),
    }
    (log_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print("\n========== EXP10 TEST RESULTS ==========")
    print(f"Test samples    : {len(test_ds)}")
    print(f"Parameters      : {total_params:,} (trainable: {trainable_params:,})")
    print(f"Test Loss       : {test_loss:.4f}")
    print(f"Accuracy        : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro Precision : {macro_precision:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")
    print("\nReports saved to:", log_dir)


if __name__ == "__main__":
    main()
