"""Stage existing baseline runs, train Test 4 variant, then evaluate fixed checkpoints."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STGCN = ROOT.parent / "stgcn"
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from experiments.exp10rerun.model_exp10rerun import InfoGCN
from research_audit.Test5_mask_correction_comparison.model_mask_correct import MaskedInfoGCN

METADATA = STGCN / "datasets/processed/metadata_split.csv"
SOURCE_RUNS = {21: "exp10rerun_seed21", 42: "exp10_seed42", 7: "exp10_seed7", 13: "exp10_seed13"}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset_class():
    path = STGCN / "experiments/common/dataset.py"
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("test5_kathakali_dataset", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.KathakaliDataset
    finally:
        sys.dont_write_bytecode = old


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id):
    s = torch.initial_seed() % (2**32)
    np.random.seed(s); random.seed(s)


def class_names():
    df = pd.read_csv(METADATA)
    return df[["label_index", "class_name"]].drop_duplicates().sort_values("label_index").class_name.tolist()


def eval_loader(model, loader, criterion, device):
    model.eval(); total, n, truth, pred = 0.0, 0, [], []
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, mask)
            loss = criterion(logits, y)
            total += loss.item() * y.size(0); n += y.size(0)
            truth.extend(y.cpu().tolist()); pred.extend(logits.argmax(1).cpu().tolist())
    return total / n, float(accuracy_score(truth, pred)), float(f1_score(truth, pred, average="macro", zero_division=0))


def stage_baseline(config):
    preflight = json.loads((OUT / "preflight.json").read_text())
    assert preflight["status"].startswith("PASS")
    expected_hash = preflight["config_sha256"]
    assert digest(OUT / "comparison_config.json") == expected_hash, "Config changed after preflight"
    baseline_dir = OUT / "baseline"
    if baseline_dir.exists():
        manifest = baseline_dir / "staged_baseline_manifest.json"
        if not manifest.exists():
            raise FileExistsError(f"Baseline staging directory exists without manifest: {baseline_dir}")
        return
    baseline_dir.mkdir()
    staged = {}
    for seed, source_name in SOURCE_RUNS.items():
        source_log = ROOT / "logs/exp10rerun" / source_name
        source_ckpt = ROOT / "runs/exp10rerun" / source_name / "best_model.pt"
        dest = baseline_dir / f"seed_{seed}"
        dest.mkdir()
        for filename in ("training_log.csv", "best_validation_metrics.json"):
            shutil.copy2(source_log / filename, dest / filename)
        shutil.copy2(source_ckpt, dest / "best_model.pt")
        staged[str(seed)] = {
            "source_run": source_name,
            "checkpoint_sha256": digest(dest / "best_model.pt"),
            "training_log_sha256": digest(dest / "training_log.csv"),
            "validation_summary_sha256": digest(dest / "best_validation_metrics.json"),
        }
    (baseline_dir / "staged_baseline_manifest.json").write_text(json.dumps(staged, indent=2) + "\n")


def train_corrected(seed, config, Dataset):
    train_cfg = config["training"]
    run_dir = OUT / "mask_correct" / f"seed_{seed}"
    if run_dir.exists():
        # Preserve a prior interrupted attempt, then start a fresh run directory.
        failed_root = OUT / "failed_attempts"
        failed_root.mkdir(exist_ok=True)
        target = failed_root / f"seed_{seed}_attempt_{len(list(failed_root.iterdir())) + 1}"
        os.replace(run_dir, target)
        print(f"Preserved prior incomplete run at {target}", flush=True)
    run_dir.mkdir(parents=True)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = class_names()
    train_ds, val_ds = Dataset("train", in_channels=3), Dataset("val", in_channels=3)
    gen = torch.Generator(); gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                              num_workers=train_cfg["num_workers"], pin_memory=device.type == "cuda",
                              worker_init_fn=seed_worker if train_cfg["num_workers"] else None, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    model_config = {"num_classes": len(names), "in_channels": 3, "num_joints": 50,
                    "base_channels": train_cfg["base_channels"], "dropout": train_cfg["dropout"],
                    "head_dropout": train_cfg["head_dropout"], "use_ib": False, "latent_dim": 128}
    model = MaskedInfoGCN(**model_config).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=train_cfg["scheduler_factor"], patience=train_cfg["scheduler_patience"])
    history_path = run_dir / "training_log.csv"
    ckpt_path = run_dir / "best_model.pt"
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1", "learning_rate"])

    best_loss, best_acc, best_f1, best_epoch, wait = float("inf"), 0.0, 0.0, 0, 0
    stopped, last_epoch = False, 0
    for epoch in range(1, train_cfg["epochs"] + 1):
        last_epoch = epoch; model.train()
        total_loss, total_n, truth, pred = 0.0, 0, [], []
        for x, y, mask in train_loader:
            x, y, mask = x.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg["gradient_clip_norm"])
            optimizer.step()
            n = y.size(0); total_loss += loss.item() * n; total_n += n
            truth.extend(y.cpu().tolist()); pred.extend(logits.argmax(1).detach().cpu().tolist())
        tr_loss = total_loss / total_n
        tr_acc = float(accuracy_score(truth, pred))
        val_loss, val_acc, val_f1 = eval_loader(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[MASK-CORRECT seed={seed}] epoch={epoch:03d}/{train_cfg['epochs']} train_loss={tr_loss:.6f} train_acc={tr_acc:.4f} val_loss={val_loss:.6f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f} lr={lr_now:.7g}", flush=True)
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, tr_acc, val_loss, val_acc, val_f1, lr_now])
        improved = val_loss < best_loss - 1e-4
        if improved:
            best_loss, best_acc, best_f1, best_epoch, wait = val_loss, val_acc, val_f1, epoch, 0
            torch.save({"experiment": "test5_mask_correct", "seed": seed, "epoch": epoch,
                        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "model_config": model_config, "best_val_loss": best_loss,
                        "val_accuracy": best_acc, "val_macro_f1": best_f1, "class_names": names,
                        "total_parameters": sum(p.numel() for p in model.parameters()),
                        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}, ckpt_path)
        else:
            wait += 1
        if wait >= train_cfg["patience"]:
            stopped = True
            print(f"[MASK-CORRECT seed={seed}] early stopping at epoch {epoch}", flush=True)
            break
    payload = {
        "variant": "mask_correct_test4", "seed": seed, "best_epoch": best_epoch, "last_epoch": last_epoch,
        "stopped_early": stopped, "selection_metric": "val_loss", "best_validation_loss": best_loss,
        "best_validation_accuracy": best_acc, "best_validation_macro_f1": best_f1,
        "epochs_configured": train_cfg["epochs"], "batch_size": train_cfg["batch_size"],
        "learning_rate": train_cfg["lr"], "weight_decay": train_cfg["weight_decay"],
        "scheduler": "plateau", "model_config": model_config,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint": str(ckpt_path), "training_log": str(history_path),
    }
    (run_dir / "best_validation_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-seeds", action="store_true", required=True)
    args = parser.parse_args()
    del args
    config = json.loads((OUT / "comparison_config.json").read_text())
    train_cfg = config["training"]
    stage_baseline(config)
    (OUT / "mask_correct").mkdir(exist_ok=True)
    (OUT / "run_environment.json").write_text(json.dumps({
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pytorch": torch.__version__,
    }, indent=2) + "\n")
    Dataset = load_dataset_class()
    summaries = []
    for seed in config["seeds"]:
        summaries.append(train_corrected(seed, config, Dataset))
    (OUT / "mask_correct" / "training_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    # Evaluation is imported and called only after every corrected run is fixed.
    from research_audit.Test5_mask_correction_comparison.evaluate_comparison import evaluate_all
    evaluate_all(config)


if __name__ == "__main__":
    main()
