#!/usr/bin/env python3
"""Validation-only Transformer tuning with a locked independent test evaluation."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from temporal_data import FeatureScaler, build_bundle, labels
from temporal_models import (
    EnhancedTimeAwareTransformerClassifier, TimeAwareTransformerClassifier,
    SequenceDataset, collate_sequences,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ProcessedData/temporal_results/transformer_validation_tuning"
DATA_DIR = ROOT / "ProcessedData/model_dataset/pre_sepsis_anchors"
ONTOLOGY_DIR = ROOT / "ProcessedData/model_dataset/ontology_enhanced"
CACHE_DIR = ROOT / "ProcessedData/model_dataset/temporal_cache"
BASE_RESULTS = ROOT / "ProcessedData/temporal_results/main"
SEED = 2026
CONFIGS = [
    {"name": "locked_baseline", "architecture": "legacy", "hidden": 128, "heads": 4,
     "layers": 3, "dropout": .15, "lr": 2e-4, "wd": 1e-4, "pooling": "last"},
    {"name": "enhanced_compact", "architecture": "enhanced", "hidden": 128, "heads": 4,
     "layers": 2, "dropout": .10, "lr": 3e-4, "wd": 1e-5, "pooling": "attention"},
    {"name": "enhanced_wide", "architecture": "enhanced", "hidden": 192, "heads": 4,
     "layers": 3, "dropout": .10, "lr": 2e-4, "wd": 1e-4, "pooling": "attention"},
    {"name": "enhanced_regularized", "architecture": "enhanced", "hidden": 192, "heads": 8,
     "layers": 2, "dropout": .20, "lr": 1e-4, "wd": 1e-4, "pooling": "last"},
]


def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed)


def make_model(config, input_dim):
    if config["architecture"] == "legacy":
        return TimeAwareTransformerClassifier(
            input_dim, config["hidden"], config["heads"], config["layers"], config["dropout"]
        )
    return EnhancedTimeAwareTransformerClassifier(
        input_dim, config["hidden"], config["heads"], config["layers"], config["dropout"],
        max_seq_len=24, pooling=config["pooling"],
    )


@torch.no_grad()
def predict(model, examples, batch_size=256):
    loader = DataLoader(SequenceDataset(examples), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_sequences, num_workers=0)
    model.eval(); out=[]
    for batch in loader:
        logits=model(batch["x"],batch["observed_mask"],batch["delta_hours"],batch["lengths"])
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def fit_config(config, train, validation, input_dim, seed):
    seed_all(seed)
    model=make_model(config,input_dim)
    y=labels(train); pos=max(int(y.sum()),1); neg=max(int((1-y).sum()),1)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],dtype=torch.float32))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=config["wd"])
    batch_size=int(config.get("batch_size",256))
    max_epochs=int(config.get("max_epochs",15))
    patience=int(config.get("patience",3))
    min_epochs=int(config.get("min_epochs",6))
    loader=DataLoader(SequenceDataset(train),batch_size=batch_size,shuffle=True,collate_fn=collate_sequences,
                      num_workers=0,generator=torch.Generator().manual_seed(seed))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode="max",factor=.5,patience=max(2,patience//2),min_lr=2e-6
    ) if config.get("lr_scheduler",False) else None
    best=-np.inf; best_state=None; best_epoch=0; stale=0; history=[]
    for epoch in range(1,max_epochs+1):
        model.train(); losses=[]
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch["x"],batch["observed_mask"],batch["delta_hours"],batch["lengths"])
            loss=criterion(logits,batch["labels"]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
            losses.append(float(loss.detach()))
        p=predict(model,validation); score=average_precision_score(labels(validation),p)
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_auprc":score,
                        "learning_rate":float(optimizer.param_groups[0]["lr"])})
        print(config["name"],"epoch",epoch,"val AUPRC",round(score,5),flush=True)
        if score > best + 1e-4:
            best=score; best_epoch=epoch; best_state=deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=patience and epoch>=min_epochs: break
        if scheduler is not None:
            scheduler.step(score)
    model.load_state_dict(best_state)
    return model,best,best_epoch,history


def metric(y,p):
    return {"auprc":average_precision_score(y,p),"auroc":roc_auc_score(y,p),
            "brier":brier_score_loss(y,p)}


def paired_ci(y,new,old,seed,reps=5000):
    rng=np.random.default_rng(seed); vals={"auprc":[],"auroc":[],"brier":[]}
    for _ in range(reps):
        idx=rng.integers(0,len(y),len(y)); yy=y[idx]
        if len(np.unique(yy))<2:continue
        a,b=metric(yy,new[idx]),metric(yy,old[idx])
        for k in vals: vals[k].append(a[k]-b[k])
    out={}
    for k,v in vals.items():
        out[f"delta_{k}"]=float(np.mean(v)); out[f"delta_{k}_low"]=float(np.quantile(v,.025)); out[f"delta_{k}_high"]=float(np.quantile(v,.975))
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True); leaderboard=[]; selected=[]
    for horizon in (6,12,24):
        task=f"pre_{horizon}h"
        bundle=build_bundle(task,"ontology",DATA_DIR,
                            ONTOLOGY_DIR,
                            ROOT/"ProcessedData/feature_schema.json",
                            CACHE_DIR,24)
        train,validation,test=(bundle.by_split(x) for x in ("train","validation","test"))
        scaler=FeatureScaler(bundle.feature_names).fit(train); scaler.transform(bundle.examples)
        candidates=[]
        for i,config in enumerate(CONFIGS):
            started=time.time(); model,score,epoch,history=fit_config(config,train,validation,len(bundle.feature_names),SEED+i)
            row={"task":task,**config,"validation_auprc":score,"best_epoch":epoch,
                 "parameter_n":sum(p.numel() for p in model.parameters()),"fit_seconds":time.time()-started}
            leaderboard.append(row); candidates.append((score,config,model,epoch,history))
        score,config,model,epoch,history=max(candidates,key=lambda x:x[0])
        p_test=predict(model,test); y_test=labels(test); stats=metric(y_test,p_test)
        old=pd.read_csv(BASE_RESULTS / "predictions" / f"{task}_ontology_transformer_test.csv")
        assert old.visit_no.tolist()==[e.visit_no for e in test]
        deltas=paired_ci(y_test,p_test,old.probability.to_numpy(),SEED+horizon)
        selected.append({"task":task,"selected_config":config["name"],"validation_auprc":score,
                         "best_epoch":epoch,**stats,**deltas})
        pd.DataFrame({"visit_no":[e.visit_no for e in test],"label":y_test,"probability":p_test}).to_csv(
            OUT/f"{task}_selected_transformer_test.csv",index=False)
        torch.save({"task":task,"config":config,"state_dict":model.state_dict(),"feature_names":bundle.feature_names,
                    "scaler":scaler.state_dict(),"best_epoch":epoch,"validation_auprc":score},
                   OUT/f"{task}_selected_transformer.pt")
    pd.DataFrame(leaderboard).to_csv(OUT/"validation_leaderboard.csv",index=False)
    pd.DataFrame(selected).to_csv(OUT/"selected_test_results.csv",index=False)
    print("\nSelected locked-test results")
    print(pd.DataFrame(selected).to_string(index=False))


if __name__=="__main__": main()
