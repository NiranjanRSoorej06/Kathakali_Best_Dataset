"""Consolidated multi-seed summary reporter for EXP10RERUN (Seeds 21, 42, 7, 13)."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

INFOGCN_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "multiseed_config.json"
LOGS_ROOT = INFOGCN_ROOT / "logs" / "exp10rerun"
RUNS_ROOT = INFOGCN_ROOT / "runs" / "exp10rerun"
DEFAULT_OUTPUT_DIR = LOGS_ROOT / "multiseed_summary"

METRIC_KEYS = [
    "best_epoch",
    "best_validation_loss",
    "best_validation_accuracy",
    "best_validation_macro_f1",
    "test_loss",
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
]


def load_run_record(run_name: str, expected_seed: int, expected_hparams: dict) -> dict:
    """Load and validate per-seed validation and canonical test artifacts read-only."""
    log_dir = LOGS_ROOT / run_name
    ckpt_path = RUNS_ROOT / run_name / "best_model.pt"
    val_path = log_dir / "best_validation_metrics.json"
    test_path = log_dir / "test_metrics.json"

    missing = [str(p) for p in (ckpt_path, val_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run '{run_name}' is incomplete; missing:\n  " + "\n  ".join(missing))

    val_metrics = json.loads(val_path.read_text(encoding="utf-8"))
    test_metrics = json.loads(test_path.read_text(encoding="utf-8"))

    if int(val_metrics["seed"]) != expected_seed:
        raise ValueError(
            f"Seed mismatch for '{run_name}': expected {expected_seed}, found {val_metrics['seed']}"
        )
    if int(val_metrics["best_epoch"]) != int(test_metrics["checkpoint_epoch"]):
        raise ValueError(
            f"Epoch mismatch for '{run_name}': val best_epoch={val_metrics['best_epoch']} "
            f"vs test checkpoint_epoch={test_metrics['checkpoint_epoch']}"
        )
    if int(test_metrics["test_samples"]) != 65:
        raise ValueError(f"Expected 65 canonical test samples for '{run_name}', found {test_metrics['test_samples']}")

    hparam_mismatches = []
    for k, expected_val in (
        ("batch_size", expected_hparams["batch_size"]),
        ("learning_rate", expected_hparams["lr"]),
        ("weight_decay", expected_hparams["weight_decay"]),
        ("scheduler", expected_hparams["scheduler"]),
        ("selection_metric", expected_hparams["selection_metric"]),
    ):
        if val_metrics.get(k) != expected_val:
            hparam_mismatches.append(f"{k}: expected {expected_val}, got {val_metrics.get(k)}")
    if hparam_mismatches:
        raise ValueError(f"Hyperparameter mismatch for '{run_name}': " + "; ".join(hparam_mismatches))

    return {
        "run_name": run_name,
        "seed": expected_seed,
        "best_epoch": int(val_metrics["best_epoch"]),
        "best_validation_loss": float(val_metrics["best_validation_loss"]),
        "best_validation_accuracy": float(val_metrics["best_validation_accuracy"]),
        "best_validation_macro_f1": float(val_metrics["best_validation_macro_f1"]),
        "test_samples": int(test_metrics["test_samples"]),
        "test_loss": float(test_metrics["test_loss"]),
        "test_accuracy": float(test_metrics["test_accuracy"]),
        "test_macro_precision": float(test_metrics["test_macro_precision"]),
        "test_macro_recall": float(test_metrics["test_macro_recall"]),
        "test_macro_f1": float(test_metrics["test_macro_f1"]),
        "test_weighted_f1": float(test_metrics["test_weighted_f1"]),
        "gpu_name": test_metrics.get("gpu_name"),
        "checkpoint": str(ckpt_path),
    }


def summarize_metrics(records: list[dict]) -> dict[str, dict[str, float]]:
    """Compute mean, sample std (ddof=1), population std (ddof=0), min, and max."""
    summary = {}
    for key in METRIC_KEYS:
        vals = np.asarray([r[key] for r in records], dtype=np.float64)
        summary[key] = {
            "mean": float(vals.mean()),
            "std_sample": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "std_pop": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return summary


def write_markdown_summary(
    output_path: Path,
    records: list[dict],
    stats: dict[str, dict[str, float]],
    missing_runs: list[dict],
) -> None:
    lines = [
        "# EXP10RERUN Multi-Seed Canonical Summary",
        "",
        f"- **Completed Seeds Included**: `{len(records)}` ({', '.join(str(r['seed']) for r in records)})",
        "",
        "## Per-Seed Results (Canonical 300/65/65 Split)",
        "| Run Name | Seed | Best Epoch | Val Loss | Val Acc | Val Macro F1 | Test Loss | Test Acc | Test Macro F1 | Test Weighted F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['run_name']} | {r['seed']} | {r['best_epoch']} | "
            f"{r['best_validation_loss']:.4f} | {r['best_validation_accuracy']:.4f} | {r['best_validation_macro_f1']:.4f} | "
            f"{r['test_loss']:.4f} | {r['test_accuracy']:.4f} | {r['test_macro_f1']:.4f} | {r['test_weighted_f1']:.4f} |"
        )

    lines.extend([
        "",
        "## Aggregate Mean ± Std (Sample `ddof=1`)",
        "| Metric | Mean | Std (ddof=1) | Std (ddof=0) | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key in METRIC_KEYS:
        s = stats[key]
        lines.append(
            f"| `{key}` | {s['mean']:.4f} | {s['std_sample']:.4f} | {s['std_pop']:.4f} | {s['min']:.4f} | {s['max']:.4f} |"
        )

    if missing_runs:
        lines.extend(["", "## Incomplete / Pending Runs"])
        for item in missing_runs:
            lines.append(f"- `{item['run_name']}` (seed `{item['seed']}`): {item['error']}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize EXP10RERUN multi-seed canonical results")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Summarize completed runs even if some configured seeds have not finished yet",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in logs/exp10rerun/multiseed_summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    expected_hparams = cfg["shared_hyperparameters"]

    records = []
    missing_runs = []
    for run in cfg["runs"]:
        try:
            records.append(load_run_record(run["run_name"], int(run["seed"]), expected_hparams))
        except FileNotFoundError as exc:
            missing_runs.append({"run_name": run["run_name"], "seed": run["seed"], "error": str(exc)})

    if missing_runs and not args.allow_partial:
        details = "\n".join(f"- {m['run_name']} (seed {m['seed']}): {m['error']}" for m in missing_runs)
        raise RuntimeError(
            "All 4 configured EXP10RERUN seeds must be trained and evaluated before generating "
            f"the final multi-seed summary (or pass --allow-partial):\n{details}"
        )

    if not records:
        raise RuntimeError("No completed EXP10RERUN runs found to summarize.")

    output_files = [
        args.output_dir / "multiseed_per_run.csv",
        args.output_dir / "multiseed_summary.json",
        args.output_dir / "multiseed_summary.md",
    ]
    existing = [str(p) for p in output_files if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite existing summary outputs:\n" + "\n".join(existing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = summarize_metrics(records)

    with open(args.output_dir / "multiseed_per_run.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "experiment": "exp10rerun",
        "config_path": str(args.config),
        "shared_hyperparameters": expected_hparams,
        "completed_runs": len(records),
        "expected_runs": len(cfg["runs"]),
        "missing_runs": missing_runs,
        "per_run": records,
        "aggregate_metrics": stats,
    }
    (args.output_dir / "multiseed_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(args.output_dir / "multiseed_summary.md", records, stats, missing_runs)

    print("EXP10RERUN Multi-Seed Summary")
    print(f"Completed runs : {len(records)} / {len(cfg['runs'])}")
    print(
        f"Test Accuracy  : {stats['test_accuracy']['mean'] * 100:.2f}% "
        f"± {stats['test_accuracy']['std_sample'] * 100:.2f}%"
    )
    print(
        f"Test Macro F1  : {stats['test_macro_f1']['mean']:.4f} "
        f"± {stats['test_macro_f1']['std_sample']:.4f}"
    )
    print(f"Summary saved  : {args.output_dir}")


if __name__ == "__main__":
    main()
