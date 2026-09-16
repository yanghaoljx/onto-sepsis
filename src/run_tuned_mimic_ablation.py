#!/usr/bin/env python3
"""External MIMIC validation of matched enhanced-Transformer transfer arms.

Set ``MIMIC_BUCKET_HOURS`` and ``MIMIC_OUTPUT_TAG`` to run an auditable
time-bucket sensitivity without overwriting the default outputs.
"""

from __future__ import annotations

import csv, json, os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from mimic_external_validation.run_temporal_external_validation import CONVERSION, derive_states
from mimic_external_validation.train_structured_derived_partial_ontology import PARTIAL_RELATIONS, PARTIAL_STATES
from temporal_data import FeatureScaler, SequenceBundle, SequenceExample, build_bundle, labels
from temporal_models import EnhancedTimeAwareTransformerClassifier, TimeAwareTransformerClassifier
from tune_time_aware_transformer import fit_config, predict
from sanitize_local_ontology_features import RANGES as LOCAL_QUALITY_RANGES

def clean_external_values(values):
    """Apply source-independent data-quality rules after unit conversion.

    The glucose upper guard mirrors MIMIC's chemistry.sql limit (10000 mg/dL)
    for both glucose sources, catching unfiltered chart-event sentinel values.
    Existing source values are not clipped to improve external performance.
    """
    cleaned={}
    for name,value in values.items():
        if not np.isfinite(value): continue
        limits=LOCAL_QUALITY_RANGES.get(name)
        if limits is not None and not limits[0] <= value <= limits[1]: continue
        if name in {'value__lab__glucose','value__vital__glucose'} and not 0 < value <= 10000/18.018: continue
        cleaned[name]=value
    return cleaned

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"ProcessedData/mimic_external_validation"
INTERNAL=Path(os.getenv("MIMIC_INTERNAL_DIR", ROOT/"ProcessedData/temporal_results/paired_partial_ontology_quality_corrected_v3"))
TUNED=ROOT/"ProcessedData/temporal_results/transformer_validation_tuning_observed_scaling"
OUT=Path(os.getenv("MIMIC_TRAINING_DIR", ROOT/"ProcessedData/temporal_results/tuned_mimic_transfer_corrected"))
TAG=os.getenv("MIMIC_OUTPUT_TAG", "_corrected")
BUCKET_HOURS=float(os.getenv("MIMIC_BUCKET_HOURS", "6"))
ABLATE_DERIVED=os.getenv("MIMIC_ABLATE_DERIVED", "0") == "1"
FORCE_OBSERVED=os.getenv("MIMIC_FORCE_OBSERVED", "0") == "1"
TABLE=ROOT/f"ProcessedData/temporal_results/final_report/source_tables/tuned_transformer_mimic_ablation{TAG}_bootstrap.csv"
PRED_TABLE=ROOT/f"ProcessedData/temporal_results/final_report/source_tables/tuned_transformer_mimic_predictions{TAG}.csv"
VARIANTS=("structured","entity","relation","partial_ontology")
SEED=20260901; MAX_LEN=24; N_BOOT=1000


def make_bundle(raw:SequenceBundle,variant:str)->SequenceBundle:
    if variant=="structured":
        # Scaling is in place; sharing raw corrupted every subsequent arm.
        from train_partial_ontology_all_models import clone
        return clone(raw,variant)
    add_states=PARTIAL_STATES if variant in ("entity","partial_ontology") else []
    add_rel=PARTIAL_RELATIONS if variant in ("relation","partial_ontology") else []
    base=raw.feature_names[:-2]; names=base+[f"state__{x}" for x in add_states]+[f"relation_count__{x}" for x in add_rel]+raw.feature_names[-2:]
    value={n:i for i,n in enumerate(base) if n.startswith("value__")}; masks={n[6:]:i for i,n in enumerate(base) if n.startswith("mask__value__")}
    examples=[]
    for old in raw.examples:
        rows=[]; observed_rows=[]; previous=set()
        for row,obs in zip(old.x,old.observed_mask):
            vals={n:float(row[i]) for n,i in value.items() if n in masks and (row[masks[n]]>0 or row[i]!=0)}
            states,rels=derive_states(vals,previous)
            added=np.asarray([float(s in states) for s in add_states]+[float(rels.get(r,0)) for r in add_rel],np.float32)
            rows.append(np.r_[row[:-2],added,row[-2:]])
            observed_rows.append(np.r_[obs[:-2],np.ones(len(added),np.float32),obs[-2:]])
            previous=states
        examples.append(SequenceExample(np.stack(rows).astype(np.float32),old.delta_hours.copy(),old.hours_from_admission.copy(),
            np.stack(observed_rows).astype(np.float32),old.label,old.split,old.patient_id,old.visit_no,old.endpoint_time))
    return SequenceBundle(raw.task,variant,names,examples,raw.endpoints)


def medians(bundle):
    out={}; train=bundle.by_split("train"); names=bundle.feature_names
    for i,n in enumerate(names):
        if not n.startswith("value__") or "mask__"+n not in names: continue
        mi=names.index("mask__"+n); chunks=[e.x[e.x[:,mi]>0,i] for e in train if np.any(e.x[:,mi]>0)]
        vals=np.concatenate(chunks) if chunks else np.asarray([],dtype=np.float32)
        out[n]=float(np.median(vals[np.isfinite(vals)])) if len(vals) else 0.0
    return out


def train_models():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"model_artifacts").mkdir(exist_ok=True)
    existing = {}
    expected = [OUT/"model_artifacts"/f"pre_{h}h_{v}_transformer.pt" for h in (6,12,24) for v in VARIANTS]
    if all(p.exists() for p in expected):
        for h in (6,12,24):
            for v in VARIANTS:
                existing[(h,v)] = torch.load(OUT/"model_artifacts"/f"pre_{h}h_{v}_transformer.pt", map_location="cpu", weights_only=False)
        return existing
    infos={}
    for hi,h in enumerate((6,12,24)):
        task=f"pre_{h}h"; tuned_config=torch.load(TUNED/f"{task}_selected_transformer.pt",map_location="cpu",weights_only=False)["config"]
        raw=build_bundle(task,"structured",ROOT/"ProcessedData/model_dataset/pre_sepsis_anchors",ROOT/"ProcessedData/model_dataset/ontology_enhanced_quality_cleaned",
                         ROOT/"ProcessedData/feature_schema.json",ROOT/"ProcessedData/model_dataset/temporal_cache_quality_cleaned",24)
        train_medians=medians(raw)
        for vi,variant in enumerate(VARIANTS):
            config=dict(tuned_config)
            bundle=make_bundle(raw,variant); train,val,test=(bundle.by_split(x) for x in ("train","validation","test"))
            scaler=FeatureScaler(bundle.feature_names).fit(train); scaler.transform(bundle.examples)
            if variant=="structured":
                ck=torch.load(INTERNAL/"model_artifacts"/f"{task}_structured_transformer.pt",map_location="cpu",weights_only=False)
                config={"architecture":"legacy", "hidden": int(ck.get("hidden_dim", 128)), "hidden_dim": int(ck.get("hidden_dim", 128))}
                model=TimeAwareTransformerClassifier(len(bundle.feature_names), hidden_dim=config["hidden"])
                model.load_state_dict(ck["state_dict"]); val_score=ck["best_validation_auprc"]; best_epoch=ck["best_epoch"]
            else:
                model,val_score,best_epoch,_=fit_config(config,train,val,len(bundle.feature_names),SEED+hi*100+vi)
            pv=predict(model,val); yv=labels(val); thresholds=np.unique(pv)[::-1]; threshold=min(thresholds,key=lambda t:abs((((pv<t)&(yv==0)).sum()/max((yv==0).sum(),1))-.90))
            ck={"task":task,"variant":variant,"config":config,"state_dict":model.state_dict(),"feature_names":bundle.feature_names,
                "input_dim":len(bundle.feature_names),"scaler":scaler.state_dict(),"training_medians":train_medians,"max_seq_len":24,
                "threshold_at_specificity_90":float(threshold),"validation_auprc":float(val_score),"best_epoch":int(best_epoch)}
            torch.save(ck,OUT/"model_artifacts"/f"{task}_{variant}_transformer.pt")
            infos[(h,variant)]=ck
    return infos


def raw_sequence(events,meta,names,training_medians, numeric_contract="v3"):
    age,icu,pred=meta; carried={"value__demographic__age":age}; previous=set(); buckets=[]; bid=None; bt=None; bv={}
    for et,vals in events:
        vals=clean_external_values(vals)
        if numeric_contract in ("v4","v5"):
            # Only approved numeric channels may generate ontology states.
            vals={n:v for n,v in vals.items() if n in names}
        if not icu <= et <= pred:
            raise ValueError("External event falls outside the causal ICU prefix")
        if bt is not None and et < bt:
            raise ValueError("External events must be sorted chronologically")
        cur=max(0,int((et-icu).total_seconds()//(BUCKET_HOURS*3600)))
        if bid is not None and cur!=bid: buckets.append((bt,bv)); bv={}
        bid=cur; bt=et; bv.update(vals)
    if bid is not None:buckets.append((bt,bv))
    if not buckets:buckets=[(pred,{})]
    rows=[]; obsrows=[]; availrows=[]; prior=None
    for bi,(et,newvals) in enumerate(buckets):
        if bi == max(0, len(buckets)-MAX_LEN): previous=set()
        carried.update(newvals); states,rels=derive_states(carried,previous); raw=np.zeros(len(names),np.float32); obs=np.ones(len(names),np.float32)
        dh=0 if prior is None else max(0,(et-prior).total_seconds()/3600); ah=max(0,(et-icu).total_seconds()/3600)
        for j,n in enumerate(names):
            if n.startswith("value__"):
                raw[j]=float(carried.get(n,training_medians.get(n,0))); obs[j]=1.0 if (n in newvals or (n=="value__demographic__age" and bi==0)) else 0.0
            elif n.startswith("mask__value__"):
                base=n[6:]; raw[j]=1.0 if (base in newvals or (base=="value__demographic__age" and bi==0)) else 0.0
            elif n.startswith("state__"): raw[j]=float(n[7:] in states)
            elif n.startswith("relation_count__"): raw[j]=float(rels.get(n[16:],0))
            elif n=="time__delta_hours": raw[j]=dh/24
            elif n=="time__hours_from_admission": raw[j]=ah/24
        rows.append(raw); obsrows.append(obs)
        availrows.append(np.asarray([float(n in carried) if n.startswith('value__') else 1. for n in names],np.float32))
        previous=states; prior=et
    deltas=[]; prior=None
    for et,_ in buckets[-MAX_LEN:]:
        # SequenceExample.delta_hours is in hours, including for GRU-D decay.
        deltas.append(0.0 if prior is None else max(0.0,(et-prior).total_seconds()/3600.0))
        prior=et
    retained = np.stack(rows[-MAX_LEN:])
    if "time__delta_hours" in names:
        retained[0,names.index("time__delta_hours")] = 0.0
    result=retained,np.stack(obsrows[-MAX_LEN:]),np.asarray(deltas,np.float32)
    return (*result,np.stack(availrows[-MAX_LEN:])) if numeric_contract in ('v4','v5') else result


def load_model(ck):
    c=ck["config"]
    input_dim = int(ck.get("input_dim") or len(ck["feature_names"]))
    if c.get("architecture") == "legacy":
        m=TimeAwareTransformerClassifier(input_dim, hidden_dim=c["hidden"])
    else:
        m=EnhancedTimeAwareTransformerClassifier(input_dim,c["hidden"],c["heads"],c["layers"],c["dropout"],24,c["pooling"])
    m.load_state_dict(ck["state_dict"]); m.eval(); return m


def infer(info,arrays):
    n=len(arrays); f=info["input_dim"]; x=np.zeros((n,MAX_LEN,f),np.float32); o=np.zeros_like(x); lengths=np.array([len(a[0]) for a in arrays],np.int64)
    scaler=FeatureScaler.from_state_dict(info["scaler"])
    d=np.zeros((n,MAX_LEN),np.float32)
    for i,(a,b,dh) in enumerate(arrays):
        x[i,:len(a)]=scaler.transform_array(a,b); o[i,:len(b)]=b; d[i,:len(dh)]=dh
    with torch.no_grad():
        p=torch.sigmoid(info["model"](torch.from_numpy(x),torch.from_numpy(o),torch.from_numpy(d),torch.from_numpy(lengths))).numpy()
    return p,lengths


def metric(y,p,t):
    pred=p>=t; pos=y==1; neg=~pos; tp=(pred&pos).sum(); fn=((~pred)&pos).sum(); tn=((~pred)&neg).sum(); fp=(pred&neg).sum()
    return {"auprc":average_precision_score(y,p),"auroc":roc_auc_score(y,p),"brier":brier_score_loss(y,p),
            "sensitivity":tp/max(tp+fn,1),"specificity":tn/max(tn+fp,1)}


def bootstrap(y,probs,thresholds,seed):
    rng=np.random.default_rng(seed); draws={(v,k):[] for v in VARIANTS for k in ("auprc","auroc","brier","sensitivity","specificity")}
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    for _ in range(N_BOOT):
        idx=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]
        for v in VARIANTS:
            m=metric(y[idx],probs[v][idx],thresholds[v])
            for k,z in m.items():draws[(v,k)].append(z)
    return draws


def main():
    ckpts=train_models(); infos={k:{**v,"model":load_model(v)} for k,v in ckpts.items()}
    candidates=pd.read_csv(DATA/"mimic_sequence_candidates.csv")
    meta={(int(r.horizon_h),int(r.stay_id)):(float(r.admission_age),datetime.fromisoformat(r.icu_intime),datetime.fromisoformat(r.prediction_time),int(r.label)) for r in candidates.itertuples()}
    pending={h:[] for h in (6,12,24)}; predictions=[]; seen=set()
    def flush(h):
        groups=pending[h]
        if not groups:return
        allp={}; lengths=None
        for v in VARIANTS:
            inf=infos[(h,v)]; arr=[raw_sequence(g[2],g[3],inf["feature_names"],inf["training_medians"]) for g in groups]
            if FORCE_OBSERVED:
                masks=np.asarray([i for i,nm in enumerate(inf["feature_names"])
                                  if nm.startswith("mask__value__")],dtype=int)
                arr=[(a.copy(),b,dh) for a,b,dh in arr]
                for a,b,dh in arr:
                    a[:,masks]=1.0
            if ABLATE_DERIVED and v != "structured":
                # Replace derived state/relation channels by their training
                # means after scaling, preserving the model and all structured
                # inputs. This is a diagnostic counterfactual, not a new model.
                mean=np.asarray(inf["scaler"]["mean"],np.float32)
                derived=np.asarray([i for i,nm in enumerate(inf["feature_names"])
                                    if nm.startswith("state__") or nm.startswith("relation_count__")],dtype=int)
                arr=[(a.copy(),b,dh) for a,b,dh in arr]
                for a,b,dh in arr:
                    a[:,derived]=mean[derived]
            allp[v],lengths=infer(inf,arr)
        for i,g in enumerate(groups):
            row={"horizon_h":h,"stay_id":g[0],"label":g[1],"sequence_length":int(lengths[i])}
            for v in VARIANTS:row[v+"_probability"]=float(allp[v][i])
            predictions.append(row);seen.add((h,g[0]))
        groups.clear()
    with (DATA/"mimic_sequence_events.csv").open(newline="") as fh:
        reader=csv.DictReader(fh); value_cols=[x for x in reader.fieldnames if x.startswith("value__")]; current=None; events=[]; etime=None; vals={}
        def finish_time():
            nonlocal vals
            if etime is not None and vals:events.append((etime,vals))
            vals={}
        def finish_group():
            if current is None:return
            finish_time(); h,stay=current; age,icu,pred,label=meta[current]; pending[h].append((stay,label,list(events),(age,icu,pred)))
            if len(pending[h])>=256:flush(h)
        for row in reader:
            key=(int(row["horizon_h"]),int(row["stay_id"])); newtime=datetime.fromisoformat(row["event_time"])
            if key!=current:finish_group();current=key;events=[];etime=None;vals={}
            if newtime!=etime:finish_time();etime=newtime
            for n in value_cols:
                if row[n]:
                    z=float(row[n])*CONVERSION.get(n,1.0)
                    if np.isfinite(z):vals[n]=z
        finish_group()
    for key,(age,icu,pred,label) in meta.items():
        if key not in seen:
            h,stay=key;pending[h].append((stay,label,[],(age,icu,pred)))
            if len(pending[h])>=256:flush(h)
    for h in pending:flush(h)
    frame=pd.DataFrame(predictions).sort_values(["horizon_h","stay_id"]);frame.to_csv(PRED_TABLE,index=False)
    rows=[]
    for h,g in frame.groupby("horizon_h"):
        y=g.label.to_numpy(int); probs={v:g[v+"_probability"].to_numpy(float) for v in VARIANTS}; thresholds={v:infos[(h,v)]["threshold_at_specificity_90"] for v in VARIANTS}
        draws=bootstrap(y,probs,thresholds,SEED+h)
        ref=metric(y,probs["structured"],thresholds["structured"])
        for v in VARIANTS:
            m=metric(y,probs[v],thresholds[v]);row={"horizon_h":h,"variant":v,"n":len(y),"positive_n":int(y.sum()),"negative_n":int((y==0).sum()),"threshold":thresholds[v],**m}
            for k in m:row[k+"_lower"],row[k+"_upper"]=np.quantile(draws[(v,k)],[.025,.975])
            for k in ("auprc","auroc","brier"):
                ds=np.asarray(draws[(v,k)])-np.asarray(draws[("structured",k)])
                row["delta_"+k]=m[k]-ref[k];row["delta_"+k+"_lower"],row["delta_"+k+"_upper"]=np.quantile(ds,[.025,.975])
            rows.append(row)
    pd.DataFrame(rows).to_csv(TABLE,index=False);print(pd.DataFrame(rows)[["horizon_h","variant","auprc","auroc","brier","delta_auprc"]].to_string(index=False))

if __name__=="__main__":main()
