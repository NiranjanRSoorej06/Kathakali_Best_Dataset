#!/usr/bin/env python3
"""Controlled forward-only checks for EXP10RERUN mask/padding behavior.

Reads the seed-21 checkpoint without saving or modifying it. No optimizer,
dataset examples, test predictions, or New40 files are used.
"""
import copy
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CHECKPOINT=ROOT/"runs/exp10rerun/exp10rerun_seed21/best_model.pt"
sys.path.insert(0,str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from experiments.exp10rerun.model_exp10rerun import InfoGCN

torch.set_num_threads(1)
torch.manual_seed(1103)


def make_model():
    checkpoint=torch.load(CHECKPOINT,map_location="cpu",weights_only=True)
    model=InfoGCN(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"],strict=True)
    return model,checkpoint


def build_padded_inputs(batch=2,valid_len=83):
    # Identical valid frames across all variants; only masked values change.
    base=torch.randn(batch,3,100,50)*0.15
    mask=torch.ones(batch,100,dtype=torch.bool)
    mask[0,valid_len:]=False
    if batch>1: mask[1,97:]=False
    keep=mask[:,None,:,None]
    zero=torch.where(keep,base,torch.zeros_like(base))
    large=torch.where(keep,base,torch.full_like(base,25.0))
    negative=torch.where(keep,base,torch.full_like(base,-25.0))
    return base,mask,zero,large,negative


def max_abs(t): return float(t.detach().abs().max().cpu())


def main():
    model,checkpoint=make_model()
    base,mask,x_zero,x_large,x_negative=build_padded_inputs()
    model.eval()
    with torch.inference_mode():
        log_zero,_,_,pool_zero=model(x_zero,mask,return_latent=True)
        log_large,_,_,pool_large=model(x_large,mask,return_latent=True)
        log_negative,_,_,pool_negative=model(x_negative,mask,return_latent=True)
    eval_large_logit_delta=max_abs(log_zero-log_large)
    eval_negative_logit_delta=max_abs(log_zero-log_negative)
    eval_large_pool_delta=max_abs(pool_zero-pool_large)
    eval_negative_pool_delta=max_abs(pool_zero-pool_negative)

    # NaNs are not neutralized by multiplication with a zero mask.
    x_nan=x_zero.clone(); x_nan[~mask[:,None,:,None].expand_as(x_nan)]=float("nan")
    with torch.inference_mode():
        log_nan=model(x_nan,mask)
    nan_case_has_nonfinite=not bool(torch.isfinite(log_nan).all())

    # Training-mode BatchNorm uses current N,T statistics. Use independent
    # in-memory copies and disable dropout so padding values are the intervention.
    valid_bn=[]; train_logits=[]
    for padded in (x_zero,x_large):
        trial=copy.deepcopy(model)
        trial.train()
        for module in trial.modules():
            if isinstance(module,nn.modules.dropout._DropoutNd): module.eval()
        captured={}
        def capture_bn(_module,_inputs,output): captured["data_bn"]=output.detach().clone()
        h=trial.data_bn.register_forward_hook(capture_bn)
        with torch.no_grad(): out=trial(padded,mask)
        h.remove()
        valid=mask[:,None,:].expand_as(captured["data_bn"])
        valid_bn.append(captured["data_bn"][valid].clone())
        train_logits.append(out.detach().clone())
        del trial
    train_valid_bn_delta=max_abs(valid_bn[0]-valid_bn[1])
    train_logit_delta=max_abs(train_logits[0]-train_logits[1])
    train_pred_changed=bool((train_logits[0].argmax(dim=1)!=train_logits[1].argmax(dim=1)).any())

    # Intervene only on invalid hidden activations after each branch's
    # pointwise BN+ReLU and before its temporal convolution. This isolates
    # hidden-state boundary leakage in eval mode; it does not modify weights.
    one_mask=mask[:1].clone(); one_x=x_zero[:1].clone()
    stages=[]; current=one_mask
    for block in model.blocks:
        stages.append(current)
        current=current[:,::block.stride]
    observed=[]; handles=[]
    for i,(block,stage_mask) in enumerate(zip(model.blocks,stages)):
        for branch_name in ("branch_d1","branch_d2"):
            relu= getattr(block.temporal,branch_name)[2]
            invalid=(~stage_mask)[:,None,:,None]
            def record(_module,_inputs,output,*,i=i,branch_name=branch_name,invalid=invalid):
                vals=output.masked_select(invalid.expand_as(output))
                observed.append({"block":i+1,"branch":branch_name,"invalid_activation_max_abs":max_abs(vals) if vals.numel() else 0.0,
                                 "invalid_activation_nonzero_count":int(torch.count_nonzero(vals).item()) if vals.numel() else 0})
            handles.append(relu.register_forward_hook(record))
    with torch.inference_mode():
        base_logits,_,_,base_pool=model(one_x,one_mask,return_latent=True)
    for h in handles: h.remove()

    # Repeat with invalid BN+ReLU outputs suppressed before temporal mixing.
    intervention_handles=[]
    for block,stage_mask in zip(model.blocks,stages):
        for branch_name in ("branch_d1","branch_d2"):
            relu=getattr(block.temporal,branch_name)[2]
            valid_weight=stage_mask[:,None,:,None].to(torch.float32)
            intervention_handles.append(relu.register_forward_hook(lambda _m,_i,o,w=valid_weight:o*w))
    with torch.inference_mode():
        masked_logits,_,_,masked_pool=model(one_x,one_mask,return_latent=True)
    for h in intervention_handles: h.remove()
    boundary_logit_delta=max_abs(base_logits-masked_logits)
    boundary_pool_delta=max_abs(base_pool-masked_pool)
    invalid_act_max=max((r["invalid_activation_max_abs"] for r in observed),default=0.0)
    invalid_act_count=sum(r["invalid_activation_nonzero_count"] for r in observed)

    # Canonical validity metadata (read-only): count rows whose entire sampled
    # 100-frame representation is marked valid.
    metadata_path=ROOT.parent/"stgcn/datasets/processed/metadata_split.csv"
    meta=pd.read_csv(metadata_path)
    valid_counts=meta["valid_frames"].astype(int)
    canonical={"clips":int(len(meta)),"all_100_valid_clips":int((valid_counts==100).sum()),
              "clips_with_fewer_than_100_valid_frames":int((valid_counts<100).sum()),
              "minimum_valid_frames":int(valid_counts.min()),"maximum_valid_frames":int(valid_counts.max())}

    report={
        "checkpoint_read_only":str(CHECKPOINT),"checkpoint_run_name":checkpoint.get("run_name"),"checkpoint_epoch":int(checkpoint["epoch"]),
        "test_type":"synthetic forward-only; no optimizer or parameter update",
        "mask_valid_lengths":mask.sum(dim=1).tolist(),"eval_finite_padding_value_change":{
            "zero_vs_plus25_max_logit_abs_delta":eval_large_logit_delta,"zero_vs_minus25_max_logit_abs_delta":eval_negative_logit_delta,
            "zero_vs_plus25_max_pooled_representation_abs_delta":eval_large_pool_delta,"zero_vs_minus25_max_pooled_representation_abs_delta":eval_negative_pool_delta,
            "interpretation":"Finite raw padded values are removed by eval-mode per-channel input normalization followed by masking; invariant in this controlled check."},
        "eval_nan_padding": {"masked_nan_produced_nonfinite_logits":nan_case_has_nonfinite,"interpretation":"Multiplication by zero does not clear NaN/Inf; input finiteness is required."},
        "train_mode_batchnorm_padding_value_change":{
            "zero_vs_plus25_max_valid_data_bn_abs_delta":train_valid_bn_delta,"zero_vs_plus25_max_logit_abs_delta":train_logit_delta,
            "predicted_class_changed_for_any_sample":train_pred_changed,
            "interpretation":"Changing only invalid finite values changes current BatchNorm1d statistics and valid normalized activations; forward-only test, no optimizer step."},
        "eval_internal_invalid_branch_activation":{
            "invalid_branch_activation_max_abs":invalid_act_max,"nonzero_invalid_branch_activation_count":invalid_act_count,
            "zero_invalid_branch_outputs_before_temporal_conv_max_logit_delta":boundary_logit_delta,
            "zero_invalid_branch_outputs_before_temporal_conv_max_pooled_representation_delta":boundary_pool_delta,
            "branch_observations":observed,
            "interpretation":"Counterfactual activation intervention tests whether learned BN+ReLU outputs at masked positions contribute through temporal convolutions to valid pooled outputs."},
        "canonical_metadata":canonical,
        "scope_limits":"Synthetic sequence only; not a dataset evaluation. Checkpoint read but not modified. No test split or New40 files read."
    }
    (HERE/"controlled_mask_results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

    labels=["Eval: +25 pad","Eval: -25 pad","Train BN: +25 pad","Eval: masked invalid\nbranch activations","Eval: NaN pad"]
    vals=[eval_large_logit_delta,eval_negative_logit_delta,train_logit_delta,boundary_logit_delta,float(nan_case_has_nonfinite)]
    fig,ax=plt.subplots(figsize=(10,4.8))
    colors=["#27823b" if v<1e-6 else "#d27a00" for v in vals]
    bars=ax.bar(labels,vals,color=colors)
    ax.set_ylabel("Max absolute logit difference (or 1 if non-finite)")
    ax.set_title("Controlled EXP10RERUN padding sensitivity checks")
    ax.grid(axis="y",alpha=.25)
    for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"{v:.3g}",ha="center",va="bottom",fontsize=9)
    fig.tight_layout(); fig.savefig(HERE/"mask_sensitivity_checks.png",dpi=170); plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
