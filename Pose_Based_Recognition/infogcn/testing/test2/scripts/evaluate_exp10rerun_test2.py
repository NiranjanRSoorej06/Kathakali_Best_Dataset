"""External GPU-enforced evaluation of trained EXP10RERUN Seed 21 on New40 inside testing/test2."""

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
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_test2 import (  # noqa: E402
    EXPECTED_CHANNELS,
    EXPECTED_FRAMES,
    EXPECTED_JOINTS,
    EXPECTED_VIDEOS,
    EXP10RERUN_CHECKPOINT_PATH,
    EXP10RERUN_LOG_PATH,
    EXP10RERUN_RUN_NAME,
    INFOGCN_ROOT,
    JOINT_INDICES,
    NEW40_MANIFEST_PATH,
    RESULTS_DIR,
    load_canonical_class_names,
    require_cuda,
)

if str(INFOGCN_ROOT) not in sys.path:
    sys.path.insert(0, str(INFOGCN_ROOT))

from experiments.exp10rerun.model_exp10rerun import InfoGCN, InfoGCNLoss  # noqa: E402


class ExternalTest2Dataset(Dataset):
    """Validated in-memory dataset wrapper for New40 preprocessed NPZ files."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        x = torch.from_numpy(sample["features"]).permute(2, 0, 1).contiguous()
        y = torch.tensor(sample["label_index"], dtype=torch.long)
        mask = torch.from_numpy(sample["frame_valid"].copy())
        return x, y, mask


def verify_checkpoint_selection(checkpoint: dict) -> dict:
    """Confirm checkpoint matches the lowest canonical validation-loss epoch in training_log.csv."""
    if not EXP10RERUN_LOG_PATH.exists():
        raise FileNotFoundError(f"EXP10RERUN training log not found: {EXP10RERUN_LOG_PATH}")
    log_df = pd.read_csv(EXP10RERUN_LOG_PATH)
    best_row = log_df.loc[log_df["val_loss"].idxmin()]
    log_best_epoch = int(best_row["epoch"])
    log_best_val_loss = float(best_row["val_loss"])
    ckpt_epoch = int(checkpoint["epoch"])
    ckpt_val_loss = float(checkpoint["best_val_loss"])

    if log_best_epoch != ckpt_epoch:
        raise ValueError(
            f"Checkpoint epoch {ckpt_epoch} does not match lowest val_loss epoch "
            f"{log_best_epoch} in {EXP10RERUN_LOG_PATH}"
        )

    return {
        "selection_criterion": "lowest canonical validation loss (never canonical test set)",
        "training_log_path": str(EXP10RERUN_LOG_PATH),
        "best_epoch": ckpt_epoch,
        "checkpoint_val_loss": ckpt_val_loss,
        "log_val_loss": log_best_val_loss,
        "checkpoint_val_accuracy": float(checkpoint["val_accuracy"]),
        "checkpoint_val_macro_f1": float(checkpoint["val_macro_f1"]),
        "seed": int(checkpoint["seed"]),
    }


def validate_and_load_samples(
    manifest_path: Path,
    class_names: list[str],
) -> tuple[list[dict], list[dict]]:
    """Validate every manifest entry and processed NPZ against canonical EXP10RERUN requirements."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    valid_samples = []
    skipped_or_invalid = []
    seen_paths = set()

    for _, row in df.iterrows():
        video_name = str(row["video_name"])
        npz_path = Path(str(row["output"]))
        seen_paths.add(npz_path.resolve())
        try:
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing NPZ file: {npz_path}")

            with np.load(npz_path, allow_pickle=False) as data:
                features = data["features"].astype(np.float32)
                frame_valid = data["frame_valid"].astype(bool)
                joint_indices = data["joint_indices"]
                label = str(data["label"].item())
                label_index = int(data["label_index"].item())
                norm_valid_frames = int(data["normalization_valid"].astype(bool).sum())

            if features.shape != (EXPECTED_FRAMES, EXPECTED_JOINTS, EXPECTED_CHANNELS):
                raise ValueError(f"Unexpected features shape {features.shape}")
            if not np.isfinite(features).all():
                raise ValueError("Non-finite values in features")
            if frame_valid.shape != (EXPECTED_FRAMES,) or not frame_valid.any():
                raise ValueError(f"Invalid frame_valid mask (valid={int(frame_valid.sum())})")
            if not np.array_equal(joint_indices, JOINT_INDICES):
                raise ValueError("50-joint subset indices do not match canonical [0..7]+[92..133]")
            if label not in class_to_index or class_to_index[label] != label_index:
                raise ValueError(f"Class mapping mismatch: label={label}, label_index={label_index}")
            if int(row["label_index"]) != label_index or str(row["label"]) != label:
                raise ValueError("Manifest label mismatch against NPZ contents")

            valid_samples.append({
                "video_name": video_name,
                "npz_path": str(npz_path),
                "features": features,
                "frame_valid": frame_valid,
                "label": label,
                "label_index": label_index,
                "valid_frames": int(frame_valid.sum()),
                "normalization_valid_frames": norm_valid_frames,
            })
        except Exception as exc:
            skipped_or_invalid.append({
                "video_name": video_name,
                "npz_path": str(npz_path),
                "reason": repr(exc),
            })

    manifest_root = manifest_path.parent
    disk_npz = {p.resolve() for p in manifest_root.rglob("*.npz")}
    unindexed = sorted(str(p) for p in (disk_npz - seen_paths))
    for path_str in unindexed:
        skipped_or_invalid.append({
            "video_name": Path(path_str).name,
            "npz_path": path_str,
            "reason": f"Present on disk under {manifest_root} but missing from {manifest_path.name}",
        })

    return valid_samples, skipped_or_invalid


def write_markdown_report(
    report_path: Path,
    metrics: dict,
    pred_rows: list[dict],
    report_dict: dict,
    class_names: list[str],
    skipped_or_invalid: list[dict],
) -> None:
    """Write a complete markdown evaluation report inside testing/test2/results."""
    correct_rows = [r for r in pred_rows if r["correct"]]
    incorrect_rows = [r for r in pred_rows if not r["correct"]]

    lines = [
        "# EXP10RERUN Seed 21 External Evaluation Report (`testing/test2`)",
        "",
        "## Execution & Environment",
        f"- **Device**: `{metrics['device']}` (`{metrics['gpu_name']}`)",
        f"- **CUDA Version**: `{metrics['cuda_version']}`",
        f"- **Run Name**: `{metrics['run_name']}` (seed `{metrics['checkpoint_selection']['seed']}`)",
        f"- **Source Manifest**: `{metrics['manifest_path']}`",
        f"- **Checkpoint**: `{metrics['checkpoint']}`",
        f"- **Checkpoint Epoch**: `{metrics['checkpoint_epoch']}` (selected via lowest canonical validation loss: `{metrics['checkpoint_selection']['checkpoint_val_loss']:.6f}`)",
        f"- **Model Parameters**: `{metrics['total_parameters']:,}` (trainable: `{metrics['trainable_parameters']:,}`)",
        f"- **Evaluated Videos**: `{metrics['evaluated_videos']}` / `{EXPECTED_VIDEOS}`",
        f"- **Skipped / Invalid Videos**: `{metrics['skipped_or_invalid_count']}`",
        "",
        "## Summary Metrics",
        f"- **Overall Accuracy**: `{metrics['accuracy']:.4f}` (`{metrics['accuracy'] * 100:.2f}%`, `{metrics['correct_count']}/{metrics['evaluated_videos']}`)",
        f"- **Macro F1 (all 41 canonical classes)**: `{metrics['macro_f1_all_41']:.4f}`",
        f"- **Macro F1 (40 represented classes)**: `{metrics['macro_f1_represented']:.4f}`",
        f"- **Weighted F1**: `{metrics['weighted_f1']:.4f}`",
        f"- **Macro Precision (41 / represented)**: `{metrics['macro_precision_all_41']:.4f}` / `{metrics['macro_precision_represented']:.4f}`",
        f"- **Macro Recall (41 / represented)**: `{metrics['macro_recall_all_41']:.4f}` / `{metrics['macro_recall_represented']:.4f}`",
        f"- **Mean Cross-Entropy Loss**: `{metrics['eval_loss']:.4f}`",
        "",
        "## Class Coverage",
        f"- **Represented Classes ({metrics['represented_class_count']})**: {', '.join(metrics['represented_classes'])}",
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

    if skipped_or_invalid:
        lines.extend(["", "## Skipped / Invalid Videos"])
        for item in skipped_or_invalid:
            lines.append(f"- `{item['video_name']}`: {item['reason']}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained EXP10RERUN Seed 21 on New40 (testing/test2)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path, default=EXP10RERUN_CHECKPOINT_PATH)
    parser.add_argument("--manifest", type=Path, default=NEW40_MANIFEST_PATH)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results in testing/test2/results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, gpu_name = require_cuda()
    print(f"Evaluation device: {device} ({gpu_name})")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"EXP10RERUN checkpoint not found: {args.checkpoint}")

    output_files = [
        RESULTS_DIR / "external_eval_metrics.json",
        RESULTS_DIR / "classification_report.csv",
        RESULTS_DIR / "classification_report.txt",
        RESULTS_DIR / "confusion_matrix.csv",
        RESULTS_DIR / "predictions.csv",
        RESULTS_DIR / "skipped_invalid_videos.json",
        RESULTS_DIR / "external_eval_report.md",
    ]
    existing = [str(p) for p in output_files if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite existing test2 results:\n" + "\n".join(existing))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    class_names = load_canonical_class_names()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_classes = checkpoint.get("class_names")
    if ckpt_classes is None or list(ckpt_classes) != class_names:
        raise ValueError("Checkpoint class_names do not match canonical metadata_split.csv")

    selection_info = verify_checkpoint_selection(checkpoint)

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

    samples, skipped_or_invalid = validate_and_load_samples(args.manifest, class_names)
    (RESULTS_DIR / "skipped_invalid_videos.json").write_text(
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

    if len(samples) != EXPECTED_VIDEOS or skipped_or_invalid:
        raise RuntimeError(
            f"External evaluation requires exactly {EXPECTED_VIDEOS} valid videos; "
            f"loaded={len(samples)}, skipped_or_invalid={len(skipped_or_invalid)}. "
            f"See {RESULTS_DIR / 'skipped_invalid_videos.json'}"
        )

    dataset = ExternalTest2Dataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

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

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

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
        y_true_arr,
        y_pred_arr,
        labels=labels_all,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_txt = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels_all,
        target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels_all)

    pd.DataFrame(report_dict).transpose().to_csv(RESULTS_DIR / "classification_report.csv")
    (RESULTS_DIR / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(RESULTS_DIR / "confusion_matrix.csv")

    pred_rows = []
    with open(RESULTS_DIR / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_name",
            "true_id",
            "true_class",
            "predicted_id",
            "predicted_class",
            "correct",
            "confidence",
            "true_class_prob",
            "valid_frames",
            "normalization_valid_frames",
            "npz_path",
        ])
        for sample, t, p, conf, tprob in zip(samples, y_true, y_pred, pred_confs, true_probs):
            is_correct = (t == p)
            row_dict = {
                "video_name": sample["video_name"],
                "true_id": t,
                "true_class": class_names[t],
                "predicted_id": p,
                "predicted_class": class_names[p],
                "correct": is_correct,
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
        "run_name": str(checkpoint.get("run_name", EXP10RERUN_RUN_NAME)),
        "evaluation_type": "external_40_video_generalization_test2",
        "device": str(device),
        "gpu_name": gpu_name,
        "cuda_version": torch.version.cuda,
        "manifest_path": str(args.manifest),
        "checkpoint": str(args.checkpoint),
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
    (RESULTS_DIR / "external_eval_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(
        RESULTS_DIR / "external_eval_report.md",
        metrics,
        pred_rows,
        report_dict,
        class_names,
        skipped_or_invalid,
    )

    print("\n========== EXP10RERUN SEED 21 EXTERNAL EVALUATION (testing/test2) ==========")
    print(f"GPU Device                : {gpu_name} ({device})")
    print(f"Checkpoint                : {args.checkpoint} (epoch {metrics['checkpoint_epoch']})")
    print(f"Evaluated Videos          : {total_samples} (skipped/invalid: {len(skipped_or_invalid)})")
    print(f"Correct / Incorrect       : {metrics['correct_count']} / {metrics['incorrect_count']}")
    print(f"Overall Accuracy          : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro F1 (41 canonical)   : {macro_f1_all:.4f}")
    print(f"Macro F1 (40 represented) : {macro_f1_rep:.4f}")
    print(f"Weighted F1               : {weighted_f1:.4f}")
    print(f"Represented / Missing     : {len(represented_classes)} / {len(missing_classes)} ({missing_classes})")
    print(f"Results saved to          : {RESULTS_DIR}")


if __name__ == "__main__":
    main()
