"""Create cross-seed tables, figures and cautious report after canonical evaluation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEEDS = [21, 42, 7, 13]
VARIANTS = ["baseline", "mask_correct"]


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def main():
    ev = json.loads((ROOT / "evaluation/all_seed_metrics_and_paired.json").read_text())
    records = []
    for seed in SEEDS:
        for variant in VARIANTS:
            records.append(ev["by_seed"][str(seed)][variant])
    metrics_df = pd.DataFrame(records)
    # Add validation-selected epoch/loss details to the comparison table.
    for idx, row in metrics_df.iterrows():
        variant = row["variant"]
        run = ROOT / variant / f"seed_{int(row['seed'])}"
        val = json.loads((run / "best_validation_metrics.json").read_text())
        for key in ("best_validation_loss", "best_validation_accuracy", "best_validation_macro_f1", "last_epoch", "stopped_early"):
            metrics_df.loc[idx, key] = val[key]
    metrics_df.to_csv(ROOT / "per_seed_test_metrics.csv", index=False)
    paired_df = pd.DataFrame([ev["paired"][str(seed)] for seed in SEEDS])
    disagreements = []
    for seed in SEEDS:
        b = pd.read_csv(ROOT / "evaluation" / f"seed_{seed}" / "baseline_predictions.csv")
        m = pd.read_csv(ROOT / "evaluation" / f"seed_{seed}" / "mask_correct_predictions.csv")
        disagreements.append(int(np.sum(b.predicted_id.to_numpy() != m.predicted_id.to_numpy())))
    paired_df["prediction_disagreements"] = disagreements
    paired_df.to_csv(ROOT / "paired_seed_comparisons.csv", index=False)

    summary = []
    for variant in VARIANTS:
        sub = metrics_df[metrics_df.variant == variant]
        row = {"variant": variant}
        for metric in ("test_loss", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "best_epoch"):
            mu, sd = mean_sd(sub[metric])
            row[f"{metric}_mean"] = mu; row[f"{metric}_sd"] = sd
        summary.append(row)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(ROOT / "cross_seed_summary.csv", index=False)

    # Pairwise metric change by seed and its small-n descriptive mean/std.
    delta_rows = []
    for seed in SEEDS:
        b = ev["by_seed"][str(seed)]["baseline"]
        m = ev["by_seed"][str(seed)]["mask_correct"]
        row = {"seed": seed}
        for metric in ("test_loss", "accuracy", "macro_f1", "weighted_f1", "best_epoch"):
            row[f"{metric}_delta_mask_minus_baseline"] = float(m[metric] - b[metric])
        delta_rows.append(row)
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(ROOT / "paired_metric_deltas.csv", index=False)

    # Per-class metrics grouped over 4 seeds. Support is repeated per seed and should match.
    class_rows = []
    for variant in VARIANTS:
        frames = []
        for seed in SEEDS:
            p = ROOT / "evaluation" / f"seed_{seed}" / f"{variant}_per_class_metrics.csv"
            df = pd.read_csv(p, index_col=0)
            df = df.loc[~df.index.isin(["accuracy", "macro avg", "weighted avg"])]
            frames.append(df)
        classes = frames[0].index
        for cls in classes:
            row = {"variant": variant, "class": cls,
                   "support": float(np.mean([float(f.loc[cls, "support"]) for f in frames]))}
            for metric in ("precision", "recall", "f1-score"):
                vals = [float(f.loc[cls, metric]) for f in frames]
                row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(vals)
            class_rows.append(row)
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(ROOT / "per_class_cross_seed_summary.csv", index=False)
    support_df = class_df[class_df.variant == "baseline"][["class", "support"]].copy()
    support_df["support"] = support_df.support.astype(int)
    support_df.to_csv(ROOT / "test_class_support.csv", index=False)
    support_stats = {
        "class_count": int(len(support_df)), "classes_with_zero_support": int((support_df.support == 0).sum()),
        "classes_with_one_support": int((support_df.support == 1).sum()),
        "minimum": int(support_df.support.min()), "median": float(support_df.support.median()),
        "maximum": int(support_df.support.max()),
    }

    # Confusion matrices aggregate counts over each of the four independently trained seeds.
    for variant in VARIANTS:
        cms = [pd.read_csv(ROOT / "evaluation" / f"seed_{s}" / f"{variant}_confusion_matrix.csv", index_col=0) for s in SEEDS]
        total_cm = sum(cms)
        total_cm.to_csv(ROOT / f"{variant}_confusion_matrix_aggregate.csv")

    # Learning curves: per-seed thin lines and cross-seed mean with available epochs.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    panels = [("train_loss", "Training loss"), ("val_loss", "Validation loss"),
              ("train_accuracy", "Training accuracy"), ("val_accuracy", "Validation accuracy")]
    colors = {"baseline": "#2864a5", "mask_correct": "#d26a24"}
    histories = {}
    for variant in VARIANTS:
        histories[variant] = []
        for seed in SEEDS:
            df = pd.read_csv(ROOT / variant / f"seed_{seed}" / "training_log.csv")
            histories[variant].append(df)
    max_epoch = max(int(df.epoch.max()) for vv in histories.values() for df in vv)
    for ax, (metric, title) in zip(axes.flat, panels):
        for variant in VARIANTS:
            arrays = np.full((len(SEEDS), max_epoch), np.nan)
            for i, df in enumerate(histories[variant]):
                vals = df[metric].to_numpy(float); arrays[i, :len(vals)] = vals
                ax.plot(df.epoch, vals, color=colors[variant], alpha=0.16, linewidth=0.8)
            x = np.arange(1, max_epoch + 1)
            count = np.sum(np.isfinite(arrays), axis=0)
            mu = np.divide(np.nansum(arrays, axis=0), count, out=np.full(max_epoch, np.nan), where=count > 0)
            variance = np.divide(np.nansum((arrays - mu[None, :]) ** 2, axis=0), count - 1,
                                 out=np.zeros(max_epoch), where=count > 1)
            sd = np.sqrt(variance)
            ax.plot(x, mu, color=colors[variant], linewidth=2, label=variant.replace("_", " "))
            ax.fill_between(x, mu - sd, mu + sd, color=colors[variant], alpha=0.13)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=0.25)
        if "loss" in metric: ax.set_ylabel("Cross-entropy loss")
        else: ax.set_ylabel("Accuracy")
    axes[0, 0].legend(); fig.suptitle("EXP10RERUN vs mask-correct: learning curves (mean ± seed SD)")
    fig.savefig(ROOT / "learning_curves.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, ["accuracy", "macro_f1", "weighted_f1"], ["Accuracy", "Macro-F1", "Weighted-F1"]):
        for variant, offset, color in [("baseline", -0.08, colors["baseline"]), ("mask_correct", 0.08, colors["mask_correct"])]:
            vals = metrics_df[metrics_df.variant == variant][metric].to_numpy(float)
            ax.scatter(np.arange(len(SEEDS)) + offset, vals, color=color, label=variant.replace("_", " "))
            ax.hlines(np.mean(vals), -0.2, len(SEEDS)-0.8, color=color, linewidth=2)
        ax.set_xticks(range(len(SEEDS)), [str(s) for s in SEEDS]); ax.set_title(title); ax.set_xlabel("Seed")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(); fig.suptitle("Canonical test metrics by seed")
    fig.savefig(ROOT / "test_metrics_by_seed.png", dpi=180); plt.close(fig)

    delta_stats = {metric: {"mean": mean_sd(delta_df[f"{metric}_delta_mask_minus_baseline"])[0],
                            "sd": mean_sd(delta_df[f"{metric}_delta_mask_minus_baseline"])[1]}
                   for metric in ("test_loss", "accuracy", "macro_f1", "weighted_f1")}
    report = [
        "# Test 5 — Controlled EXP10RERUN vs Mask-Correct Comparison", "",
        "## Design and isolation", "",
        "The canonical 430-video split was retained unchanged (300 train / 65 validation / 65 test; 41 classes). Baseline training logs and selected checkpoints were copied from the four already-completed EXP10RERUN runs whose configuration matched the pre-registered Test 5 settings; baseline runs were not repeated. The original experiment files were not modified. Four fresh mask-correct runs used the Test 4 implementation and the same seeds, batch order policy, optimizer, scheduler, epoch cap, early stopping, selection rule, and preprocessing. No augmentation was used. Checkpoints were selected by validation loss only.", "",
        "The configuration and isolation preflight passed before the fresh runs: all 430 feature paths existed; train/validation split contents were finite with 100 valid frames per example. Afterward, a separate initialization check confirmed all 207 mapped parameter/buffer tensors were identical for each of the four seeds. Test feature contents were not opened during preflight. The canonical test split was loaded only after all four mask-correct checkpoints were fixed. New40 remained sealed. Both baseline and corrected evaluation used the same H100 PCIe MIG 2g.20gb device class; exact MIG instance identity for historical baseline runs was not recorded.", "",
        "## Results", "",
        "Cross-seed mean ± sample SD (four seeds) is in `cross_seed_summary.csv`; seed-level scores and paired changes are in `per_seed_test_metrics.csv`, `paired_metric_deltas.csv`, and `paired_seed_comparisons.csv`. Per-class scores are in `per_class_cross_seed_summary.csv`; per-seed and aggregate confusion matrices are provided as CSV files. Learning curves and test-score plots are included.", "",
        "| Metric | Baseline mean ± SD | Mask-correct mean ± SD | Paired change mean ± SD |", "|---|---:|---:|---:|",
    ]
    base_row = summary_df.set_index("variant").loc["baseline"]
    mask_row = summary_df.set_index("variant").loc["mask_correct"]
    for metric in ("test_loss", "accuracy", "macro_f1", "weighted_f1"):
        b = f"{base_row[metric+'_mean']:.4f} ± {base_row[metric+'_sd']:.4f}"
        m = f"{mask_row[metric+'_mean']:.4f} ± {mask_row[metric+'_sd']:.4f}"
        d = f"{delta_stats[metric]['mean']:+.4f} ± {delta_stats[metric]['sd']:.4f}"
        report.append(f"| {metric} | {b} | {m} | {d} |")
    report += ["", "Validation-selected checkpoint summary:", "",
               "| Variant | Best validation loss, mean ± SD | Best validation accuracy, mean ± SD | Best epoch, mean ± SD |", "|---|---:|---:|---:|"]
    for variant in VARIANTS:
        sub = metrics_df[metrics_df.variant == variant]
        vals = []
        for metric in ("best_validation_loss", "best_validation_accuracy", "best_epoch"):
            mu, sd = mean_sd(sub[metric]); vals.append(f"{mu:.4f} ± {sd:.4f}")
        report.append(f"| {variant} | {vals[0]} | {vals[1]} | {vals[2]} |")
    total_disagreements = int(sum(disagreements))
    report += ["", "## Interpretation and limits", "",
        "Interpret differences descriptively. Four seeds and 65 test clips provide limited uncertainty resolution; test examples are repeated across seeds and the same split is shared. Exact paired McNemar binomial p-values are supplied per seed but have low power and are not a substitute for independent recordings or group-held-out evaluation. Per-class metrics are particularly variable where test support is small.", "",
        f"The 65-video test set covers {support_stats['class_count']} classes; per-class support ranges {support_stats['minimum']}–{support_stats['maximum']} (median {support_stats['median']:.1f}); {support_stats['classes_with_one_support']} classes have one example and {support_stats['classes_with_zero_support']} have none. See `test_class_support.csv`. Across the four seed-paired canonical test prediction files, the models disagreed on {total_disagreements} of 260 seed-by-clip predictions; see `paired_seed_comparisons.csv` for per-seed counts. Equal labels and aggregate scores on this split do not establish equal confidence calibration or cross-session robustness.", "",
        "All canonical clips had 100 valid frames. Therefore the mask correction is not expected to materially change canonical computations: there are no invalid temporal positions for it to exclude. The paired predictions were identical for every seed, while test loss varied slightly; validation curves also closely overlap. Differences in loss and selected epoch may reflect floating-point/reduction order or training dynamics of the custom valid-only BatchNorm implementation, and should be interpreted against seed variation. This experiment does not establish behavior on variable-length external clips or improve source/session independence.", "",
        "The first run attempt failed on the default MIG instance with an NVML/CUDA caching-allocator internal assertion during the first training forward, before any checkpoint was written. That incomplete log is preserved in `failed_attempts/`. The run then completed on a less occupied MIG instance of the same device class with all hyperparameters unchanged.", "",
        "The mask-correct model has a distinct state-dict layout for normalization; old checkpoints are not directly interchangeable. No New40 evaluation or per-video inspection was done.", "",
        "## Reproducibility", "",
        "Configuration: `comparison_config.json`; preflight record: `preflight.json`; initialization check: `initialization_parity.json`; commands: `commands_and_paths.md`; per-run logs/checkpoints: `baseline/` and `mask_correct/`; evaluation artifacts: `evaluation/`. The Test 4 model source is copied byte-for-byte into `model_mask_correct.py` here; baseline InfoGCN and the shared dataset loader are imported read-only.", ""]
    (ROOT / "Test5_comparison_report.md").write_text("\n".join(report))
    print(f"Wrote summary report and plots under {ROOT}")


if __name__ == "__main__":
    main()
