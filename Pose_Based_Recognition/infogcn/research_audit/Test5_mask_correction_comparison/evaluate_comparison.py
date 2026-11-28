"""One-time canonical test evaluation after all paired checkpoints are fixed."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STGCN = ROOT.parent / "stgcn"
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader

from experiments.exp10rerun.model_exp10rerun import InfoGCN
from research_audit.Test5_mask_correction_comparison.model_mask_correct import MaskedInfoGCN


def dataset_class():
    path = STGCN / "experiments/common/dataset.py"
    old = sys.dont_write_bytecode; sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("test5_eval_dataset", path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod.KathakaliDataset
    finally:
        sys.dont_write_bytecode = old


def evaluate_one(model_cls, ckpt_path, ds, batch_size, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = model_cls(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    y_true, y_pred, loss_sum, total = [], [], 0.0, 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, mask)
            loss_sum += criterion(logits, y).item() * y.size(0); total += y.size(0)
            y_true.extend(y.cpu().tolist()); y_pred.extend(logits.argmax(1).cpu().tolist())
    return ckpt, np.asarray(y_true), np.asarray(y_pred), loss_sum / total


def evaluate_all(config):
    eval_root = OUT / "evaluation"
    if eval_root.exists():
        raise FileExistsError(f"Refusing to overwrite {eval_root}")
    eval_root.mkdir()
    # Guard: no test loader is created until every validation-selected checkpoint exists.
    for seed in config["seeds"]:
        for variant in ("baseline", "mask_correct"):
            p = OUT / variant / f"seed_{seed}" / "best_model.pt"
            if not p.exists():
                raise FileNotFoundError(f"Evaluation blocked; missing fixed checkpoint: {p}")
        for variant in ("baseline", "mask_correct"):
            p = OUT / variant / f"seed_{seed}" / "best_validation_metrics.json"
            summary = json.loads(p.read_text())
            if summary["selection_metric"] != "val_loss":
                raise ValueError(f"Unexpected selection metric in {p}")

    Dataset = dataset_class()
    test_ds = Dataset("test", in_channels=3)
    names = (pd.read_csv(STGCN / "datasets/processed/metadata_split.csv")
             [["label_index", "class_name"]].drop_duplicates().sort_values("label_index").class_name.tolist())
    labels = list(range(len(names)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = {"test_samples": len(test_ds), "class_names": names, "by_seed": {}, "paired": {}}

    for seed in config["seeds"]:
        output["by_seed"][str(seed)] = {}
        pair_preds = {}
        common_truth = None
        for variant, folder, cls in (("baseline", "baseline", InfoGCN),
                                     ("mask_correct", "mask_correct", MaskedInfoGCN)):
            run = OUT / folder / f"seed_{seed}"
            ckpt, truth, pred, loss = evaluate_one(cls, run / "best_model.pt", test_ds,
                                                   config["training"]["batch_size"], device)
            if common_truth is not None and not np.array_equal(common_truth, truth):
                raise RuntimeError("Test loader order/labels differ across variants")
            common_truth = truth
            pair_preds[variant] = pred
            report = classification_report(truth, pred, labels=labels, target_names=names,
                                           output_dict=True, zero_division=0)
            cm = confusion_matrix(truth, pred, labels=labels)
            run_eval = eval_root / f"seed_{seed}"
            run_eval.mkdir(exist_ok=True)
            pd.DataFrame(report).transpose().to_csv(run_eval / f"{variant}_per_class_metrics.csv")
            pd.DataFrame(cm, index=names, columns=names).to_csv(run_eval / f"{variant}_confusion_matrix.csv")
            with open(run_eval / f"{variant}_predictions.csv", "w", newline="") as f:
                writer = csv.writer(f); writer.writerow(["test_row", "true_id", "predicted_id", "correct"])
                writer.writerows((i, int(t), int(p), bool(t == p)) for i, (t, p) in enumerate(zip(truth, pred)))
            metrics = {
                "variant": variant, "seed": seed, "best_epoch": int(ckpt["epoch"]),
                "test_samples": len(test_ds), "test_loss": float(loss),
                "accuracy": float(accuracy_score(truth, pred)),
                "macro_precision": float(precision_score(truth, pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(truth, pred, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(truth, pred, average="weighted", zero_division=0)),
                "parameters": int(sum(p.numel() for p in cls(**ckpt["model_config"]).parameters())),
            }
            (run_eval / f"{variant}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            output["by_seed"][str(seed)][variant] = metrics

        base_correct = pair_preds["baseline"] == common_truth
        mask_correct = pair_preds["mask_correct"] == common_truth
        base_only = int(np.sum(base_correct & ~mask_correct))
        mask_only = int(np.sum(~base_correct & mask_correct))
        pvalue = float(binomtest(min(base_only, mask_only), base_only + mask_only, 0.5).pvalue) if base_only + mask_only else 1.0
        paired = {
            "seed": seed,
            "baseline_only_correct": base_only,
            "mask_correct_only_correct": mask_only,
            "both_correct": int(np.sum(base_correct & mask_correct)),
            "both_wrong": int(np.sum(~base_correct & ~mask_correct)),
            "accuracy_delta_mask_minus_baseline": float(accuracy_score(common_truth, pair_preds["mask_correct"]) - accuracy_score(common_truth, pair_preds["baseline"])),
            "macro_f1_delta_mask_minus_baseline": float(f1_score(common_truth, pair_preds["mask_correct"], average="macro", zero_division=0) - f1_score(common_truth, pair_preds["baseline"], average="macro", zero_division=0)),
            "exact_mcnemar_binomial_p": pvalue,
        }
        output["paired"][str(seed)] = paired
    (eval_root / "all_seed_metrics_and_paired.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"evaluation_complete": True, "test_samples": len(test_ds), "device": str(device),
                      "output": str(eval_root)}, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit("Call evaluate_all through run_comparison.py after all runs are fixed.")
