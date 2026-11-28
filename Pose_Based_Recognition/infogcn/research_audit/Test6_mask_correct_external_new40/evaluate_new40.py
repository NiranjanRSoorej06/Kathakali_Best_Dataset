"""Stage 2/3: frozen Test 5 mask-correct New40 evaluation and aggregate comparison."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TEST5 = ROOT / "research_audit" / "Test5_mask_correction_comparison"
TEST2 = ROOT / "testing" / "test2"
BASELINE_RESULTS = ROOT / "testing" / "test2_multiseed" / "results"
MANIFEST = ROOT.parent / "stgcn" / "testing" / "new40" / "processed" / "manifest.csv"
SEEDS = [21, 42, 7, 13]
METRICS = ["accuracy", "macro_precision_all_41", "macro_recall_all_41", "macro_f1_all_41",
           "macro_f1_represented", "weighted_f1", "eval_loss"]

sys.path.insert(0, str(TEST2 / "scripts"))
sys.path.insert(0, str(ROOT))
from common_test2 import EXPECTED_VIDEOS, load_canonical_class_names, require_cuda  # noqa: E402
from evaluate_exp10rerun_test2 import ExternalTest2Dataset, validate_and_load_samples  # noqa: E402
from research_audit.Test5_mask_correction_comparison.model_mask_correct import MaskedInfoGCN  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def feature_bundle_digest(samples: list[dict]) -> str:
    h = hashlib.sha256()
    for sample in samples:
        path = Path(sample["npz_path"])
        h.update(path.name.encode())
        h.update(digest(path).encode())
    return h.hexdigest()


def scores(truth: np.ndarray, pred: np.ndarray, class_names: list[str], loss: float) -> dict:
    all_ids = np.arange(len(class_names))
    represented = np.unique(truth)
    return {
        "evaluated_videos": int(len(truth)), "correct_count": int((truth == pred).sum()),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_precision_all_41": float(precision_score(truth, pred, labels=all_ids, average="macro", zero_division=0)),
        "macro_recall_all_41": float(recall_score(truth, pred, labels=all_ids, average="macro", zero_division=0)),
        "macro_f1_all_41": float(f1_score(truth, pred, labels=all_ids, average="macro", zero_division=0)),
        "macro_f1_represented": float(f1_score(truth, pred, labels=represented, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, pred, labels=all_ids, average="weighted", zero_division=0)),
        "eval_loss": float(loss), "represented_class_count": int(len(represented)),
        "zero_support_classes": [class_names[i] for i in all_ids if i not in set(represented.tolist())],
    }


def main() -> None:
    preflight_path = OUT / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("inference_performed"):
        raise RuntimeError("Stage 1 preflight has not passed cleanly")
    device, gpu_name = require_cuda()
    class_names = load_canonical_class_names()
    samples, invalid = validate_and_load_samples(MANIFEST, class_names)
    if len(samples) != EXPECTED_VIDEOS or invalid:
        raise RuntimeError(f"New40 integrity changed after preflight: {len(samples)} samples, {len(invalid)} invalid")
    if digest(MANIFEST) != preflight["data"]["manifest_sha256"]:
        raise RuntimeError("New40 manifest changed after preflight")
    if feature_bundle_digest(samples) != preflight["data"]["feature_bundle_sha256"]:
        raise RuntimeError("New40 feature NPZ bundle changed after preflight")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_seed").mkdir(exist_ok=True)
    loader = DataLoader(ExternalTest2Dataset(samples), batch_size=16, shuffle=False,
                        num_workers=0, pin_memory=True)
    truth = np.asarray([s["label_index"] for s in samples], dtype=np.int64)
    video_names = [s["video_name"] for s in samples]
    baseline_by_seed: dict[int, dict] = {}
    mask_by_seed: dict[int, dict] = {}
    paired_rows = []
    per_seed_rows = []
    integrity = {"preflight_sha256": digest(preflight_path), "manifest_sha256": digest(MANIFEST),
                 "evaluated_video_count": len(samples), "video_name_order_sha256": preflight["data"]["video_names_sha256"],
                 "individual_video_predictions_written": False, "seeds": {}}

    for seed in SEEDS:
        baseline_dir = BASELINE_RESULTS / f"exp10_seed{seed}"
        if seed == 21:
            baseline_dir = ROOT / "testing" / "test2" / "results"
        base_metrics = json.loads((baseline_dir / "external_eval_metrics.json").read_text(encoding="utf-8"))
        base_df = pd.read_csv(baseline_dir / "predictions.csv").set_index("video_name").reindex(video_names)
        base_pred = base_df["predicted_id"].astype(int).to_numpy()
        if not np.array_equal(base_df["true_id"].astype(int).to_numpy(), truth):
            raise RuntimeError(f"Baseline seed {seed} labels do not align with New40")
        baseline_by_seed[seed] = {"pred": base_pred, "metrics": base_metrics}

        ckpt_path = TEST5 / "mask_correct" / f"seed_{seed}" / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = MaskedInfoGCN(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        pred_parts, loss_sum, n_total = [], 0.0, 0
        with torch.inference_mode():
            for x, y, mask in loader:
                x, y, mask = x.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
                logits = model(x, mask)
                loss_sum += float(criterion(logits, y).item()) * int(y.size(0))
                n_total += int(y.size(0))
                pred_parts.append(logits.argmax(dim=1).cpu().numpy())
        pred = np.concatenate(pred_parts).astype(np.int64)
        loss = loss_sum / n_total
        metrics = scores(truth, pred, class_names, loss)
        metrics.update({"seed": seed, "checkpoint": str(ckpt_path), "checkpoint_sha256": digest(ckpt_path),
                        "checkpoint_epoch": int(ckpt["epoch"]), "device": str(device), "gpu_name": gpu_name,
                        "parameter_count": sum(p.numel() for p in model.parameters()),
                        "loss_definition": "mean per-batch CrossEntropyLoss, weighted by batch sample count; use_ib=False"})
        mask_by_seed[seed] = {"pred": pred, "metrics": metrics}
        (OUT / "per_seed" / f"seed_{seed}_mask_correct_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        row = {"seed": seed, "baseline_checkpoint": base_metrics.get("checkpoint"),
               "mask_correct_checkpoint": str(ckpt_path), "baseline_prediction_disagreement_count": int((base_pred != pred).sum())}
        for metric in METRICS:
            base_key = {"macro_precision_all_41": "macro_precision_all_41", "macro_recall_all_41": "macro_recall_all_41"}.get(metric, metric)
            mask_value = metrics[metric]
            base_value = float(base_metrics[base_key])
            row[f"baseline_{metric}"] = base_value
            row[f"mask_correct_{metric}"] = mask_value
            row[f"delta_{metric}"] = mask_value - base_value
        paired_rows.append(row)
        per_seed_rows.extend([{"seed": seed, "variant": "EXP10RERUN_baseline", **{m: float(base_metrics[m]) for m in METRICS}},
                              {"seed": seed, "variant": "mask_correct", **{m: float(metrics[m]) for m in METRICS}}])
        integrity["seeds"][str(seed)] = {"checkpoint_sha256": metrics["checkpoint_sha256"],
                                          "same_video_order": True, "same_labels": True,
                                          "baseline_prediction_disagreement_count": row["baseline_prediction_disagreement_count"],
                                          "mask_correct_prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest()}
        del model, ckpt

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(OUT / "per_seed_metrics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(OUT / "paired_comparison.csv", index=False)
    aggregate_rows = []
    for variant in ("EXP10RERUN_baseline", "mask_correct"):
        subset = per_seed_df[per_seed_df.variant == variant]
        for metric in METRICS:
            vals = subset[metric].to_numpy(dtype=float)
            aggregate_rows.append({"variant": variant, "metric": metric, "mean": vals.mean(),
                                   "std_sample_ddof1": vals.std(ddof=1), "std_population_ddof0": vals.std(ddof=0)})
    for metric in METRICS:
        vals = np.asarray([r[f"delta_{metric}"] for r in paired_rows], dtype=float)
        aggregate_rows.append({"variant": "paired_delta_mask_minus_baseline", "metric": metric,
                               "mean": vals.mean(), "std_sample_ddof1": vals.std(ddof=1),
                               "std_population_ddof0": vals.std(ddof=0)})
    pd.DataFrame(aggregate_rows).to_csv(OUT / "aggregate_summary.csv", index=False)

    # Pooled (4 seeds x 40 videos) class metrics and confusion matrices; no video-level rows are saved.
    aggregate_predictions = {}
    for variant, source in (("baseline", baseline_by_seed), ("mask_correct", mask_by_seed)):
        aggregate_predictions[variant] = np.concatenate([source[s]["pred"] for s in SEEDS])
        pooled_truth = np.tile(truth, len(SEEDS))
        cm = confusion_matrix(pooled_truth, aggregate_predictions[variant], labels=np.arange(len(class_names)))
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(OUT / f"confusion_matrix_{variant}_pooled.csv")
        p = precision_score(pooled_truth, aggregate_predictions[variant], labels=np.arange(len(class_names)), average=None, zero_division=0)
        r = recall_score(pooled_truth, aggregate_predictions[variant], labels=np.arange(len(class_names)), average=None, zero_division=0)
        f = f1_score(pooled_truth, aggregate_predictions[variant], labels=np.arange(len(class_names)), average=None, zero_division=0)
        support = np.bincount(pooled_truth, minlength=len(class_names))
        pd.DataFrame({"class_name": class_names, "support": support, "zero_support": support == 0,
                      "precision": p, "recall": r, "f1": f}).to_csv(OUT / f"per_class_metrics_{variant}_pooled.csv", index=False)
        for seed in SEEDS:
            pred = source[seed]["pred"]
            cm_seed = confusion_matrix(truth, pred, labels=np.arange(len(class_names)))
            pd.DataFrame(cm_seed, index=class_names, columns=class_names).to_csv(OUT / "per_seed" / f"seed_{seed}_{variant}_confusion_matrix.csv")

    pooled_truth = np.tile(truth, len(SEEDS))
    pooled_baseline_loss = float(np.mean([baseline_by_seed[s]["metrics"]["eval_loss"] for s in SEEDS]))
    pooled_mask_loss = float(np.mean([mask_by_seed[s]["metrics"]["eval_loss"] for s in SEEDS]))
    pooled_metrics = {
        "pooled_overall_baseline": scores(pooled_truth, aggregate_predictions["baseline"], class_names, pooled_baseline_loss),
        "pooled_overall_mask_correct": scores(pooled_truth, aggregate_predictions["mask_correct"], class_names, pooled_mask_loss),
    }
    pooled_metrics["pooled_paired_delta_mask_minus_baseline"] = {
        key: (pooled_metrics["pooled_overall_mask_correct"][key] - pooled_metrics["pooled_overall_baseline"][key])
        for key in METRICS
    }
    pd.DataFrame([{"variant": variant, **vals} for variant, vals in pooled_metrics.items()]).to_csv(
        OUT / "overall_metrics_pooled_160.csv", index=False)
    for variant, vals in pooled_metrics.items():
        for metric in METRICS:
            aggregate_rows.append({"variant": variant, "metric": metric, "mean": vals[metric],
                                   "std_sample_ddof1": np.nan, "std_population_ddof0": np.nan})
    pd.DataFrame(aggregate_rows).to_csv(OUT / "aggregate_summary.csv", index=False)

    integrity["overall_prediction_disagreement_count_across_seed_video_pairs"] = int(sum(
        (baseline_by_seed[s]["pred"] != mask_by_seed[s]["pred"]).sum() for s in SEEDS))
    integrity["baseline_variant"] = "existing EXP10RERUN New40 per-seed evaluations; predictions aligned by video_name; sources preserved"
    integrity["pooled_confusion_matrix_observations_per_variant"] = int(len(SEEDS) * EXPECTED_VIDEOS)
    (OUT / "integrity_report.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "EVALUATION_COMPLETE", "videos_per_seed": len(samples), "seeds": SEEDS,
                      "aggregate_summary": str(OUT / "aggregate_summary.csv"),
                      "disagreement_count": integrity["overall_prediction_disagreement_count_across_seed_video_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
