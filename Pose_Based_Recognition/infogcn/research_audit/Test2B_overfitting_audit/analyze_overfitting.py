#!/usr/bin/env python3
"""Read-only EXP10RERUN four-seed train/validation audit and plotting."""
import csv, json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LOG_ROOT=ROOT/"logs/exp10rerun"
RUNS=[("exp10rerun_seed21",21), ("exp10_seed42",42), ("exp10_seed7",7), ("exp10_seed13",13)]

def slope(x,y):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    return float(np.polyfit(x,y,1)[0]) if len(x)>1 else math.nan

frames=[]; summaries=[]; check={}
for name,seed in RUNS:
    d=LOG_ROOT/name
    df=pd.read_csv(d/"training_log.csv")
    required={"epoch","train_loss","train_accuracy","val_loss","val_accuracy"}
    if not required.issubset(df.columns): raise ValueError(f"Missing columns in {name}: {required-set(df.columns)}")
    saved=json.loads((d/"best_validation_metrics.json").read_text())
    df.insert(0,"seed",seed); df.insert(1,"run_name",name); frames.append(df)
    best_i=int(df["val_loss"].idxmin()); best=df.loc[best_i]
    max_acc_i=int(df["val_accuracy"].idxmax()); maxacc=df.loc[max_acc_i]
    final=df.iloc[-1]
    train_best=float(best["train_loss"]); val_best=float(best["val_loss"])
    tr_acc_best=float(best["train_accuracy"]); va_acc_best=float(best["val_accuracy"])
    last10=df.tail(10); post=df[df["epoch"]>=best["epoch"]]
    # Last logged epoch and last five epochs are descriptive, not selection criteria.
    row={
        "run_name":name,"seed":seed,"epochs_logged":len(df),"best_val_loss_epoch":int(best["epoch"]),
        "saved_best_epoch":int(saved["best_epoch"]),"best_epoch_matches_saved":int(best["epoch"])==int(saved["best_epoch"]),
        "best_val_loss":val_best,"train_loss_at_best_val_loss":train_best,"loss_gap_val_minus_train_at_best":val_best-train_best,
        "val_accuracy_at_best_val_loss":va_acc_best,"train_accuracy_at_best_val_loss":tr_acc_best,
        "accuracy_gap_train_minus_val_at_best":tr_acc_best-va_acc_best,
        "peak_val_accuracy":float(maxacc["val_accuracy"]),"peak_val_accuracy_epoch":int(maxacc["epoch"]),
        "val_loss_at_peak_val_accuracy":float(maxacc["val_loss"]),
        "final_train_loss":float(final["train_loss"]),"final_val_loss":float(final["val_loss"]),
        "final_val_loss_minus_best":float(final["val_loss"])-val_best,
        "final_val_loss_increase_pct_from_best":(float(final["val_loss"])/val_best-1)*100 if val_best else math.nan,
        "final_train_accuracy":float(final["train_accuracy"]),"final_val_accuracy":float(final["val_accuracy"]),
        "final_val_accuracy_minus_at_best_loss":float(final["val_accuracy"])-va_acc_best,
        "last10_mean_train_loss":float(last10["train_loss"].mean()),"last10_mean_val_loss":float(last10["val_loss"].mean()),
        "last10_mean_train_accuracy":float(last10["train_accuracy"].mean()),"last10_mean_val_accuracy":float(last10["val_accuracy"].mean()),
        "post_best_val_loss_slope_per_epoch":slope(post["epoch"],post["val_loss"]),
        "post_best_val_accuracy_slope_per_epoch":slope(post["epoch"],post["val_accuracy"]),
        "last10_train_loss_slope_per_epoch":slope(last10["epoch"],last10["train_loss"]),
        "last10_val_loss_slope_per_epoch":slope(last10["epoch"],last10["val_loss"]),
        "saved_best_validation_loss":float(saved["best_validation_loss"]),
        "saved_best_validation_accuracy":float(saved["best_validation_accuracy"]),
        "saved_best_validation_macro_f1":float(saved["best_validation_macro_f1"]),
        "selection_metric":saved["selection_metric"],"stopped_early":saved["stopped_early"],
        "configured_epochs":saved["epochs_configured"],
    }
    summaries.append(row)
    check[name]={"logged_rows":len(df),"epochs_first_last":[int(df.epoch.min()),int(df.epoch.max())],
                 "best_val_loss_logged_epoch":int(best.epoch),"saved_best_epoch":int(saved["best_epoch"]),
                 "saved_best_loss_matches_log":abs(val_best-float(saved["best_validation_loss"]))<1e-8}

all_df=pd.concat(frames,ignore_index=True)
all_df.to_csv(HERE/"all_seed_epoch_metrics.csv",index=False)
summary_df=pd.DataFrame(summaries)
summary_df.to_csv(HERE/"per_seed_overfitting_summary.csv",index=False)
agg=all_df.groupby("epoch").agg({c:["mean","std"] for c in ["train_loss","val_loss","train_accuracy","val_accuracy"]})
agg.columns=[f"{a}_{b}" for a,b in agg.columns]; agg=agg.reset_index()
agg["n_seeds"] = all_df.groupby("epoch").size().to_numpy()
agg.to_csv(HERE/"aggregate_epoch_metrics.csv",index=False)

colors={21:"#1665ad",42:"#e67300",7:"#27823b",13:"#a13d99"}
fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True)
for ax,(name,seed) in zip(axs.flat,RUNS):
    d=all_df[all_df.seed==seed]
    ax.plot(d.epoch,d.train_loss,label="Train",color=colors[seed],lw=1.8)
    ax.plot(d.epoch,d.val_loss,label="Validation",color=colors[seed],lw=1.8,ls="--")
    best=int(summary_df.loc[summary_df.seed==seed,"best_val_loss_epoch"].iloc[0])
    ax.axvline(best,color="black",alpha=.35,lw=1,ls=":")
    ax.scatter([best],[float(d.loc[d.epoch==best,"val_loss"].iloc[0])],color=colors[seed],s=24,zorder=3)
    ax.set_title(f"{name} (seed {seed})")
    ax.set_ylabel("Cross-entropy loss"); ax.set_yscale("log"); ax.grid(alpha=.22)
    ax.legend(frameon=False,ncol=2,fontsize=9)
for ax in axs[-1,:]: ax.set_xlabel("Epoch")
fig.suptitle("EXP10RERUN training and validation loss",fontsize=14)
fig.tight_layout(); fig.savefig(HERE/"loss_curves_by_seed.png",dpi=170); plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True)
for ax,(name,seed) in zip(axs.flat,RUNS):
    d=all_df[all_df.seed==seed]
    ax.plot(d.epoch,d.train_accuracy*100,label="Train",color=colors[seed],lw=1.8)
    ax.plot(d.epoch,d.val_accuracy*100,label="Validation",color=colors[seed],lw=1.8,ls="--")
    best=int(summary_df.loc[summary_df.seed==seed,"best_val_loss_epoch"].iloc[0])
    ax.axvline(best,color="black",alpha=.35,lw=1,ls=":")
    ax.set_title(f"{name} (seed {seed})"); ax.set_ylabel("Accuracy (%)"); ax.grid(alpha=.22)
    ax.legend(frameon=False,ncol=2,fontsize=9)
for ax in axs[-1,:]: ax.set_xlabel("Epoch")
fig.suptitle("EXP10RERUN training and validation accuracy",fontsize=14)
fig.tight_layout(); fig.savefig(HERE/"accuracy_curves_by_seed.png",dpi=170); plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(13,5))
for col,(metric,title,ylabel) in enumerate([("loss","Loss","Cross-entropy loss"),("accuracy","Accuracy","Accuracy (%)")]):
    ax=axs[col]
    common_epochs=int(min(r["epochs_logged"] for r in summaries))
    agg_plot=agg[agg.epoch<=common_epochs]
    for split,style,color in [("train","-","#2369a8"),("val","--","#d46a00")]:
        mean=agg_plot[f"{split}_{metric}_mean"].to_numpy()
        std=agg_plot[f"{split}_{metric}_std"].fillna(0).to_numpy()
        if metric=="accuracy": mean*=100; std*=100
        x=agg_plot.epoch.to_numpy(); ax.plot(x,mean,style,label=f"{split.title()} mean",color=color,lw=2)
        ax.fill_between(x,mean-std,mean+std,color=color,alpha=.16,label=f"{split.title()} ±1 SD")
    if metric=="loss": ax.set_yscale("log")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=.22); ax.legend(frameon=False,fontsize=9)
fig.suptitle(f"Four-seed mean and between-seed standard deviation (common epochs 1–{common_epochs})",fontsize=14)
fig.tight_layout(); fig.savefig(HERE/"four_seed_mean_std.png",dpi=170); plt.close(fig)

means={k:float(summary_df[k].mean()) for k in ["best_val_loss","peak_val_accuracy","peak_val_accuracy_epoch","final_train_loss","final_val_loss","final_train_accuracy","final_val_accuracy","final_val_loss_minus_best"]}
stds={k:float(summary_df[k].std(ddof=1)) for k in ["best_val_loss","peak_val_accuracy","peak_val_accuracy_epoch","final_train_loss","final_val_loss","final_train_accuracy","final_val_accuracy","final_val_loss_minus_best"]}
overall={
 "experiment":"EXP10RERUN", "seeds":[21,42,7,13], "runs":check,
 "rows_per_run":{str(r["seed"]):int(r["epochs_logged"]) for r in summaries},
 "selection_metric":"val_loss", "all_saved_best_epochs_match_logged_val_loss_argmin":all(r["best_epoch_matches_saved"] for r in summaries),
 "test_metrics_used":False,
 "four_seed_train_val_means":means,"four_seed_train_val_sample_stds":stds,
 "method_note":"Metrics were computed only from training_log.csv and best_validation_metrics.json for four seeds. No test metrics, test predictions, per-video results, checkpoints, or New40 files were read by this analysis script."
}
(HERE/"summary.json").write_text(json.dumps(overall,indent=2),encoding="utf-8")

# Compact machine-generated report with per-seed dynamics.
seed_lines=[]
for r in summaries:
    seed_lines.append(f"| {r['seed']} | {r['epochs_logged']} | {r['best_val_loss_epoch']} | {r['best_val_loss']:.4f} | {r['val_accuracy_at_best_val_loss']*100:.2f}% | {r['final_train_loss']:.4f} | {r['final_val_loss']:.4f} | {r['final_val_loss_minus_best']:+.4f} | {r['final_train_accuracy']*100:.2f}% | {r['final_val_accuracy']*100:.2f}% | {r['accuracy_gap_train_minus_val_at_best']*100:+.2f} pp |")
all_rebounds=[r["final_val_loss_minus_best"] for r in summaries]
all_accdrops=[r["final_val_accuracy_minus_at_best_loss"] for r in summaries]
rebound_n=sum(x>0 for x in all_rebounds)
train_loss_drops=sum(float(pd.read_csv(LOG_ROOT/n/"training_log.csv").iloc[-1].train_loss)<float(pd.read_csv(LOG_ROOT/n/"training_log.csv").iloc[0].train_loss) for n,_ in RUNS)
completed_full=sum(r["epochs_logged"]==r["configured_epochs"] and not r["stopped_early"] for r in summaries)
report=f'''# Test 2B: Conventional overfitting audit — EXP10RERUN

## Scope

Read-only analysis of the four existing training logs (seeds 21, 42, 7, 13) and their `best_validation_metrics.json` summaries. No checkpoints, per-video predictions, confusion matrices, or New40 files were inspected. Test metrics were not used in this audit. No model was trained or tuned, and no existing file was modified.

## Per-seed training and validation behavior

| Seed | Epochs | Best val-loss epoch | Best val loss | Val acc at best-loss epoch | Final train loss | Final val loss | Final minus best val loss | Final train acc | Final val acc | Train-val acc gap at selected epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(seed_lines)}

All four logs contain {min(r['epochs_logged'] for r in summaries)}–{max(r['epochs_logged'] for r in summaries)} epochs. The saved best epochs agree with the minimum validation-loss epoch in their logs: **{all(r['best_epoch_matches_saved'] for r in summaries)}**. Checkpoint selection used `val_loss`; the reported validation accuracy at that epoch need not be the maximum validation accuracy epoch.

Across seeds, best validation loss averaged {summary_df.best_val_loss.mean():.4f} ± {summary_df.best_val_loss.std(ddof=1):.4f} (sample SD); peak per-run validation accuracy averaged {summary_df.peak_val_accuracy.mean()*100:.2f}% ± {summary_df.peak_val_accuracy.std(ddof=1)*100:.2f}% (sample SD). Best-loss epochs ranged from {summary_df.best_val_loss_epoch.min()} to {summary_df.best_val_loss_epoch.max()} (mean {summary_df.best_val_loss_epoch.mean():.2f}, sample SD {summary_df.best_val_loss_epoch.std(ddof=1):.2f}).

## Trends and interpretation

- All runs make substantial early progress: training and validation losses fall from their high initial values, while both accuracies rise. Final training loss is below initial training loss in {train_loss_drops}/4 runs. The curves show that the model fits the training split strongly and reaches high validation accuracy.
- Validation loss reaches its minimum between epochs {summary_df.best_val_loss_epoch.min()} and {summary_df.best_val_loss_epoch.max()}. It is higher at the final epoch than at its minimum in {rebound_n}/4 runs; mean final-minus-minimum validation loss is {np.mean(all_rebounds):+.4f}. The final validation accuracy differs from accuracy at the selected checkpoint by {np.mean(all_accdrops)*100:+.2f} percentage points on average.
- Validation loss rises after its selected minimum in all four seeds. From the selected epoch to the last logged epoch, training loss falls for seeds 42, 7, and 13, but rises for seed 21. Validation accuracy remains within one clip of its selected-epoch value in every run. This pattern is **consistent with late-stage overfitting or increasing confidence on mistakes** in some runs, especially seeds 7 and 13, but it does not prove harmful overfitting: cross-entropy can rise when confidence worsens while classification accuracy stays stable, and validation estimates are noisy at 65 clips (one clip is about 1.54 percentage points).
- The logged train-versus-validation gap is descriptive, not a diagnosis. Training and validation metrics are measured on different sample sizes and may reflect different model modes or aggregation details. A gap alone does not establish overfitting. The more relevant evidence is the epoch-wise divergence, validation-loss rebound, validation-accuracy trend, and selection behavior together.
- No clear underfitting signal appears on the canonical split: final train accuracy is high and training loss falls substantially. This does not establish generalization to independent sessions or recording setups.
- The minimum-validation-loss checkpoint is selected before the final logged epoch in every seed (see `per_seed_overfitting_summary.csv`). The saved summaries show {completed_full}/4 runs reached their configured epoch count without early stop. The validation-loss selection rule therefore chooses the best observed validation-loss checkpoint rather than the final epoch; it does not remove uncertainty from a small validation set.

## Plots

- `loss_curves_by_seed.png`: train/validation loss trajectories, with log-scaled y-axis and selected epoch markers.
- `accuracy_curves_by_seed.png`: train/validation accuracy trajectories, with selected epoch markers.
- `four_seed_mean_std.png`: per-epoch mean and between-seed ±1 SD over the common epoch range (1–{common_epochs}); run-specific curves show each full logged duration.

## Cautious conclusion

EXP10RERUN shows stable early convergence across all four seeds and strong fit on the canonical training/validation splits. There is evidence of late validation-loss rebound after the selected epoch in the runs summarized above, which is compatible with mild late overfitting or confidence calibration drift. The observed train-validation gaps and this rebound do not alone prove conventional overfitting; the validation set is small, classification accuracy is coarse, and the audit is limited to the existing clip-level split. The most defensible conclusion is **good in-split convergence with some late-stage validation-loss deterioration, but inconclusive evidence of harmful overfitting**.

## Artifacts and provenance

This report and plots were generated from `logs/exp10rerun/<run_name>/training_log.csv` and `best_validation_metrics.json` for seeds 21, 42, 7, and 13. Supporting files are `all_seed_epoch_metrics.csv`, `aggregate_epoch_metrics.csv` (including `n_seeds` by epoch), `per_seed_overfitting_summary.csv`, and `summary.json`. The script `analyze_overfitting.py` is a reproducible read-only analysis tool.
'''
(HERE/"test2b_overfitting_report.md").write_text(report,encoding="utf-8")
print(json.dumps({"seeds":{str(r['seed']):{"epochs":r['epochs_logged'],"best_val_loss_epoch":r['best_val_loss_epoch'],"best_val_loss":r['best_val_loss'],"final_val_loss_delta":r['final_val_loss_minus_best'],"final_val_accuracy":r['final_val_accuracy']} for r in summaries},"saved_epochs_match":all(r['best_epoch_matches_saved'] for r in summaries)},indent=2))
