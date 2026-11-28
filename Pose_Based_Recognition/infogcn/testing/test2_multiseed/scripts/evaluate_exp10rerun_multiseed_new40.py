"""GPU-based external New40 evaluation for EXP10RERUN Seeds 42, 7, 13 and 4-seed consolidation."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

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

SCRIPT_DIR = Path(__file__).resolve().parent
MULTISEED_ROOT = SCRIPT_DIR.parent
TESTING_ROOT = MULTISEED_ROOT.parent
INFOGCN_ROOT = TESTING_ROOT.parent
TEST2_SCRIPTS_DIR = TESTING_ROOT / "test2" / "scripts"

for p in (str(TEST2_SCRIPTS_DIR), str(INFOGCN_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common_test2 import (  # noqa: E402
    EXPECTED_VIDEOS,
    NEW40_MANIFEST_PATH,
    load_canonical_class_names,
    require_cuda,
)
from evaluate_exp10rerun_test2 import (  # noqa: E402
    ExternalTest2Dataset,
    validate_and_load_samples,
)
from experiments.exp10rerun.model_exp10rerun import InfoGCN, InfoGCNLoss  # noqa: E402

CONFIG_PATH = INFOGCN_ROOT / "experiments" / "exp10rerun" / "multiseed_config.json"
SEED21_METRICS_PATH = TESTING_ROOT / "test2" / "results" / "external_eval_metrics.json"
DEFAULT_RESULTS_ROOT = MULTISEED_ROOT / "results"

SUMMARY_METRICS = [
    "accuracy",
    "correct_count",
    "macro_f1_all_41",
    "macro_f1_represented",
    "weighted_f1",
    "eval_loss",
    "macro_precision_all_41",
    "macro_recall_all_41",
    "macro_precision_represented",
    "macro_recall_represented",
]


def verify_run_checkpoint_selection(checkpoint: dict, run_name: str, expected_seed: int) -> dict:
    """Confirm checkpoint matches lowest canonical validation-loss epoch in its own training_log.csv."""
    log_path = INFOGCN_ROOT / "logs" / "exp10rerun" / run_name / "training_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found for '{run_name}': {log_path}")

    log_df = pd.read_csv(log_path)
    best_row = log_df.loc[log_df["val_loss"].idxmin()]
    log_best_epoch = int(best_row["epoch"])
    log_best_val_loss = float(best_row["val_loss"])
    ckpt_epoch = int(checkpoint["epoch"])
    ckpt_val_loss = float(checkpoint["best_val_loss"])
    ckpt_seed = int(checkpoint["seed"])

    if ckpt_seed != expected_seed:
        raise ValueError(f"Seed mismatch for '{run_name}': expected {expected_seed}, got {ckpt_seed}")
    if log_best_epoch != ckpt_epoch:
        raise ValueError(
            f"Checkpoint epoch {ckpt_epoch} != lowest val_loss epoch {log_best_epoch} in {log_path}"
        )

    return {
        "selection_criterion": "lowest canonical validation loss (never canonical test set or New40)",
        "training_log_path": str(log_path),
        "best_epoch": ckpt_epoch,
        "checkpoint_val_loss": ckpt_val_loss,
        "log_val_loss": log_best_val_loss,
        "checkpoint_val_accuracy": float(checkpoint["val_accuracy"]),
        "checkpoint_val_macro_f1": float(checkpoint["val_macro_f1"]),
        "seed": ckpt_seed,
    }


def write_seed_report(
    report_path: Path,
    metrics: dict,
    pred_rows: list[dict],
    report_dict: dict,
    class_names: list[str],
) -> None:
    """Write per-seed external New40 evaluation markdown report."""
    seed = metrics["checkpoint_selection"]["seed"]
    correct_rows = [r for r in pred_rows if r["correct"]]
    incorrect_rows = [r for r in pred_rows if not r["correct"]]

    lines = [
        f"# EXP10RERUN Seed {seed} External New40 Evaluation Report (`{metrics['run_name']}`)",
        "",
        "## Execution & Environment",
        f"- **Device**: `{metrics['device']}` (`{metrics['gpu_name']}`)",
        f"- **CUDA Version**: `{metrics['cuda_version']}`",
        f"- **Run Name**: `{metrics['run_name']}` (seed `{seed}`)",
        f"- **Source Manifest**: `{metrics['manifest_path']}`",
        f"- **Checkpoint**: `{metrics['checkpoint']}`",
        f"- **Checkpoint Epoch**: `{metrics['checkpoint_epoch']}` (lowest canonical val loss: `{metrics['checkpoint_selection']['checkpoint_val_loss']:.6f}`)",
        f"- **Model Parameters**: `{metrics['total_parameters']:,}` (trainable: `{metrics['trainable_parameters']:,}`)",
        f"- **Evaluated Videos**: `{metrics['evaluated_videos']}` / `{EXPECTED_VIDEOS}` (skipped/invalid: `{metrics['skipped_or_invalid_count']}`)",
        "",
        "## Summary Metrics (New40 External)",
        f"- **Overall Accuracy**: `{metrics['accuracy']:.4f}` (`{metrics['accuracy'] * 100:.2f}%`, `{metrics['correct_count']}/{metrics['evaluated_videos']}`)",
        f"- **Macro F1 (all 41 canonical classes)**: `{metrics['macro_f1_all_41']:.4f}`",
        f"- **Macro F1 (40 represented classes)**: `{metrics['macro_f1_represented']:.4f}`",
        f"- **Weighted F1**: `{metrics['weighted_f1']:.4f}`",
        f"- **Macro Precision (41 / represented)**: `{metrics['macro_precision_all_41']:.4f}` / `{metrics['macro_precision_represented']:.4f}`",
        f"- **Macro Recall (41 / represented)**: `{metrics['macro_recall_all_41']:.4f}` / `{metrics['macro_recall_represented']:.4f}`",
        f"- **Mean Cross-Entropy Loss**: `{metrics['eval_loss']:.4f}`",
        "",
        "## Class Coverage",
        f"- **Represented Classes**: `{metrics['represented_class_count']}`",
        f"- **Missing Classes ({metrics['missing_class_count']})**: {', '.join(metrics['missing_classes']) if metrics['missing_classes'] else 'None'}",
        "",
        "## Per-Class Precision, Recall, and F1",
        "| Class | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls in class_names:
        row = report_dict[cls]
        lines.append(
            f"| {cls} | {int(row['support'])} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} |"
        )

    lines.extend([
        "",
        f"## Incorrect Predictions ({len(incorrect_rows)})",
        "| Video | True Class | Predicted Class | Confidence | True Class Prob |",
        "|---|---|---|---:|---:|",
    ])
    for r in incorrect_rows:
        lines.append(
            f"| {r['video_name']} | {r['true_class']} | {r['predicted_class']} | "
            f"{r['confidence']:.4f} | {r['true_class_prob']:.4f} |"
        )

    lines.extend([
        "",
        f"## Correct Predictions ({len(correct_rows)})",
        "| Video | Class | Confidence |",
        "|---|---|---:|",
    ])
    for r in correct_rows:
        lines.append(f"| {r['video_name']} | {r['true_class']} | {r['confidence']:.4f} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_single_run(
    run_name: str,
    seed: int,
    samples: list[dict],
    skipped_or_invalid: list[dict],
    class_names: list[str],
    manifest_path: Path,
    results_dir: Path,
    device: torch.device,
    gpu_name: str,
    batch_size: int,
    overwrite: bool,
) -> dict:
    """Evaluate one trained EXP10RERUN checkpoint on New40 and write isolated artifacts."""
    ckpt_path = INFOGCN_ROOT / "runs" / "exp10rerun" / run_name / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for '{run_name}': {ckpt_path}")

    output_files = [
        results_dir / "external_eval_metrics.json",
        results_dir / "classification_report.csv",
        results_dir / "classification_report.txt",
        results_dir / "confusion_matrix.csv",
        results_dir / "predictions.csv",
        results_dir / "skipped_invalid_videos.json",
        results_dir / "external_eval_report.md",
    ]
    existing = [str(p) for p in output_files if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing results in {results_dir}:\n" + "\n".join(existing))

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "skipped_invalid_videos.json").write_text(
        json.dumps(
            {
                "expected_videos": EXPECTED_VIDEOS,
                "valid_loaded": len(samples),
                "skipped_or_invalid_count": len(skipped_or_invalid),
                "skipped_or_invalid": skipped_or_invalid,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_classes = checkpoint.get("class_names")
    if ckpt_classes is None or list(ckpt_classes) != class_names:
        raise ValueError(f"Checkpoint class_names mismatch for '{run_name}'")

    selection_info = verify_run_checkpoint_selection(checkpoint, run_name, seed)

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
    if not next(model.parameters()).is_cuda:
        raise RuntimeError("Model parameters are not on CUDA device.")

    dataset = ExternalTest2Dataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    criterion = InfoGCNLoss(use_ib=bool(model_config.get("use_ib", False))).to(device)

    total_loss = 0.0
    total_samples = 0
    y_true, y_pred, pred_confs, true_probs = [], [], [], []

    with torch.inference_mode():
        for x, y, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if not (x.is_cuda and y.is_cuda and mask.is_cuda):
                raise RuntimeError("Evaluation tensors are not on CUDA device.")

            logits, mu, logvar, z = model(x, mask, return_latent=True)
            loss = criterion(logits, y, mu=mu, logvar=logvar, z=z)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            pred_confs.extend(probs.gather(1, preds.unsqueeze(1)).squeeze(1).cpu().numpy().tolist())
            true_probs.extend(probs.gather(1, y.unsqueeze(1)).squeeze(1).cpu().numpy().tolist())

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels_all = list(range(len(class_names)))
    represented_ids = sorted(set(y_true))
    represented_classes = [class_names[i] for i in represented_ids]
    missing_classes = [class_names[i] for i in labels_all if i not in set(represented_ids)]

    eval_loss = float(total_loss / total_samples)
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_precision_all = float(precision_score(y_true_arr, y_pred_arr, labels=labels_all, average="macro", zero_division=0))
    macro_recall_all = float(recall_score(y_true_arr, y_pred_arr, labels=labels_all, average="macro", zero_division=0))
    macro_f1_all = float(f1_score(y_true_arr, y_pred_arr, labels=labels_all, average="macro", zero_division=0))
    macro_precision_rep = float(precision_score(y_true_arr, y_pred_arr, labels=represented_ids, average="macro", zero_division=0))
    macro_recall_rep = float(recall_score(y_true_arr, y_pred_arr, labels=represented_ids, average="macro", zero_division=0))
    macro_f1_rep = float(f1_score(y_true_arr, y_pred_arr, labels=represented_ids, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels_all, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_true_arr, y_pred_arr, labels=labels_all, target_names=class_names, output_dict=True, zero_division=0
    )
    report_txt = classification_report(
        y_true_arr, y_pred_arr, labels=labels_all, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels_all)

    pd.DataFrame(report_dict).transpose().to_csv(results_dir / "classification_report.csv")
    (results_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(results_dir / "confusion_matrix.csv")

    pred_rows = []
    with open(results_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_name", "true_id", "true_class", "predicted_id", "predicted_class",
            "correct", "confidence", "true_class_prob", "valid_frames", "normalization_valid_frames", "npz_path",
        ])
        for sample, t, p, conf, tprob in zip(samples, y_true, y_pred, pred_confs, true_probs):
            row_dict = {
                "video_name": sample["video_name"],
                "true_id": t,
                "true_class": class_names[t],
                "predicted_id": p,
                "predicted_class": class_names[p],
                "correct": (t == p),
                "confidence": float(conf),
                "true_class_prob": float(tprob),
                "valid_frames": sample["valid_frames"],
                "normalization_valid_frames": sample["normalization_valid_frames"],
                "npz_path": sample["npz_path"],
            }
            pred_rows.append(row_dict)
            writer.writerow(list(row_dict.values()))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics = {
        "experiment": "exp10rerun",
        "run_name": run_name,
        "evaluation_type": "external_40_video_generalization_test2_protocol",
        "device": str(device),
        "gpu_name": gpu_name,
        "cuda_version": torch.version.cuda,
        "manifest_path": str(manifest_path),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection": selection_info,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "evaluated_videos": total_samples,
        "skipped_or_invalid_count": len(skipped_or_invalid),
        "correct_count": int((y_true_arr == y_pred_arr).sum()),
        "incorrect_count": int((y_true_arr != y_pred_arr).sum()),
        "eval_loss": eval_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1_all,
        "macro_f1_all_41": macro_f1_all,
        "macro_precision_all_41": macro_precision_all,
        "macro_recall_all_41": macro_recall_all,
        "macro_f1_represented": macro_f1_rep,
        "macro_precision_represented": macro_precision_rep,
        "macro_recall_represented": macro_recall_rep,
        "weighted_f1": weighted_f1,
        "represented_class_count": len(represented_classes),
        "represented_classes": represented_classes,
        "missing_class_count": len(missing_classes),
        "missing_classes": missing_classes,
    }
    (results_dir / "external_eval_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    write_seed_report(results_dir / "external_eval_report.md", metrics, pred_rows, report_dict, class_names)
    return metrics


def summarize_new40_multiseed(records: list[dict], output_dir: Path, overwrite: bool) -> None:
    """Consolidate New40 external evaluation metrics across all 4 seeds."""
    output_files = [
        output_dir / "new40_multiseed_per_run.csv",
        output_dir / "new40_multiseed_summary.json",
        output_dir / "new40_multiseed_summary.md",
    ]
    existing = [str(p) for p in output_files if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing summary in {output_dir}:\n" + "\n".join(existing))

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for key in SUMMARY_METRICS:
        vals = np.asarray([r[key] for r in records], dtype=np.float64)
        stats[key] = {
            "mean": float(vals.mean()),
            "std_sample": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "std_pop": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    with open(output_dir / "new40_multiseed_per_run.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "experiment": "exp10rerun",
        "evaluation_type": "external_new40_multiseed_summary",
        "manifest_path": str(NEW40_MANIFEST_PATH),
        "completed_runs": len(records),
        "per_run": records,
        "aggregate_metrics": stats,
    }
    (output_dir / "new40_multiseed_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# EXP10RERUN Multi-Seed External New40 Summary (Seeds 21, 42, 7, 13)",
        "",
        "## Per-Seed New40 External Results (40 Videos, 40 Represented / 41 Canonical Classes)",
        "| Run Name | Seed | Best Epoch | Correct / 40 | Accuracy | Macro F1 (41) | Macro F1 (40 Rep.) | Weighted F1 | Eval Loss | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['run_name']} | {r['seed']} | {r['checkpoint_epoch']} | "
            f"{r['correct_count']}/{r['evaluated_videos']} | {r['accuracy'] * 100:.2f}% | "
            f"{r['macro_f1_all_41']:.4f} | {r['macro_f1_represented']:.4f} | "
            f"{r['weighted_f1']:.4f} | {r['eval_loss']:.4f} | `{r['metrics_path']}` |"
        )

    lines.extend([
        "",
        "## Aggregate Mean ± Std (Sample `ddof=1`)",
        "| Metric | Mean | Std (ddof=1) | Std (ddof=0) | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key in SUMMARY_METRICS:
        s = stats[key]
        lines.append(
            f"| `{key}` | {s['mean']:.4f} | {s['std_sample']:.4f} | {s['std_pop']:.4f} | {s['min']:.4f} | {s['max']:.4f} |"
        )

    (output_dir / "new40_multiseed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EXP10RERUN Seeds 42, 7, 13 on New40 and summarize all 4 seeds")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--manifest", type=Path, default=NEW40_MANIFEST_PATH)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, gpu_name = require_cuda()
    print(f"Evaluation device: {device} ({gpu_name})")

    if not SEED21_METRICS_PATH.exists():
        raise FileNotFoundError(f"Existing Seed 21 New40 metrics not found: {SEED21_METRICS_PATH}")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    class_names = load_canonical_class_names()
    samples, skipped_or_invalid = validate_and_load_samples(args.manifest, class_names)
    if len(samples) != EXPECTED_VIDEOS or skipped_or_invalid:
        raise RuntimeError(
            f"External evaluation requires {EXPECTED_VIDEOS} valid videos; "
            f"loaded={len(samples)}, skipped_or_invalid={len(skipped_or_invalid)}"
        )

    seed21_metrics = json.loads(SEED21_METRICS_PATH.read_text(encoding="utf-8"))
    all_records = [{
        "run_name": seed21_metrics["run_name"],
        "seed": int(seed21_metrics["checkpoint_selection"]["seed"]),
        "checkpoint_epoch": int(seed21_metrics["checkpoint_epoch"]),
        "evaluated_videos": int(seed21_metrics["evaluated_videos"]),
        "skipped_or_invalid_count": int(seed21_metrics["skipped_or_invalid_count"]),
        "correct_count": int(seed21_metrics["correct_count"]),
        "incorrect_count": int(seed21_metrics["incorrect_count"]),
        "accuracy": float(seed21_metrics["accuracy"]),
        "macro_f1_all_41": float(seed21_metrics["macro_f1_all_41"]),
        "macro_f1_represented": float(seed21_metrics["macro_f1_represented"]),
        "weighted_f1": float(seed21_metrics["weighted_f1"]),
        "eval_loss": float(seed21_metrics["eval_loss"]),
        "macro_precision_all_41": float(seed21_metrics["macro_precision_all_41"]),
        "macro_recall_all_41": float(seed21_metrics["macro_recall_all_41"]),
        "macro_precision_represented": float(seed21_metrics["macro_precision_represented"]),
        "macro_recall_represented": float(seed21_metrics["macro_recall_represented"]),
        "checkpoint": seed21_metrics["checkpoint"],
        "metrics_path": str(SEED21_METRICS_PATH),
    }]

    for run in cfg["runs"]:
        run_name = run["run_name"]
        seed = int(run["seed"])
        if run_name == "exp10rerun_seed21":
            continue
        run_dir = args.results_root / run_name
        metrics = evaluate_single_run(
            run_name=run_name,
            seed=seed,
            samples=samples,
            skipped_or_invalid=skipped_or_invalid,
            class_names=class_names,
            manifest_path=args.manifest,
            results_dir=run_dir,
            device=device,
            gpu_name=gpu_name,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        print(
            f"[{run_name} | seed={seed}] epoch={metrics['checkpoint_epoch']} | "
            f"acc={metrics['accuracy']*100:.2f}% ({metrics['correct_count']}/{metrics['evaluated_videos']}) | "
            f"macro_f1_41={metrics['macro_f1_all_41']:.4f} | macro_f1_rep={metrics['macro_f1_represented']:.4f} | "
            f"weighted_f1={metrics['weighted_f1']:.4f}"
        )
        all_records.append({
            "run_name": metrics["run_name"],
            "seed": seed,
            "checkpoint_epoch": int(metrics["checkpoint_epoch"]),
            "evaluated_videos": int(metrics["evaluated_videos"]),
            "skipped_or_invalid_count": int(metrics["skipped_or_invalid_count"]),
            "correct_count": int(metrics["correct_count"]),
            "incorrect_count": int(metrics["incorrect_count"]),
            "accuracy": float(metrics["accuracy"]),
            "macro_f1_all_41": float(metrics["macro_f1_all_41"]),
            "macro_f1_represented": float(metrics["macro_f1_represented"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "eval_loss": float(metrics["eval_loss"]),
            "macro_precision_all_41": float(metrics["macro_precision_all_41"]),
            "macro_recall_all_41": float(metrics["macro_recall_all_41"]),
            "macro_precision_represented": float(metrics["macro_precision_represented"]),
            "macro_recall_represented": float(metrics["macro_recall_represented"]),
            "checkpoint": metrics["checkpoint"],
            "metrics_path": str(run_dir / "external_eval_metrics.json"),
        })

    summarize_new40_multiseed(all_records, args.results_root / "multiseed_summary", overwrite=args.overwrite)
    print(f"Consolidated New40 summary saved to: {args.results_root / 'multiseed_summary'}")


if __name__ == "__main__":
    main()
