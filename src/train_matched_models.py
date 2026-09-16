"""Matched structured/partial-ontology training, selected on local validation only."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import argparse
import json
import pickle
import fcntl
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from temporal_data import FeatureScaler, build_bundle, labels, metadata_rows
from run_temporal_experiments import train_sequence, train_tabular, seed_everything, _select_threshold_at_specificity, calculate_metrics
from mimic_external_validation.train_structured_derived_partial_ontology import make_partial_bundle
from train_partial_ontology_all_models import clone, DATA, ONTO, CACHE, SCHEMA, MODELS, ROOT

OUT=ROOT/'ProcessedData/temporal_results/paired_partial_ontology_quality_corrected_v5'

def train_job(job):
    bundle_path,model,out_path,seed,epochs,*extra=job
    version=extra[0] if extra else 4
    out=Path(out_path); torch.set_num_threads(1); seed_everything(seed)
    with Path(bundle_path).open('rb') as f:bundle,scaler_state=pickle.load(f)
    task,variant=bundle.task,bundle.variant; stem=f'{task}_{variant}_{model}'
    train,val,test=(bundle.by_split(s) for s in ('train','validation','test'))
    ext='.joblib' if model in MODELS[:3] else '.pt'
    path=out/'model_artifacts'/f'{stem}{ext}'
    protocol={'task':task,'variant':variant,'feature_names':bundle.feature_names,
        'scaler':scaler_state,'max_seq_len':24,'preprocessing_version':version,
        'mapping_scope':'numeric structured-derived partial ontology','training_seed':seed,'epochs':epochs}
    with (out/'logs'/f'{stem}.log').open('w') as log,redirect_stdout(log):
        if model in MODELS[:3]:
            probs,fit=train_tabular(model,bundle,train,val,test,seed,'last',path)
        else:
            probs,fit=train_sequence(model,train,val,test,torch.device('cpu'),epochs=epochs,
                batch_size=256,learning_rate=2e-4,weight_decay=1e-4,hidden_dim=128,
                num_workers=0,seed=seed,save_path=path,artifact_metadata=protocol)
    thresholds={k:_select_threshold_at_specificity(labels(val),probs['validation'],k/100) for k in (90,95)}
    records=[]
    for split,ex in [('train',train),('validation',val),('test',test)]:
        p=probs[split]
        records.append({'task':task,'model':model,'variant':variant,'split':split,
            'threshold':thresholds[90],**calculate_metrics(labels(ex),p,thresholds),**fit})
        prediction=pd.DataFrame(metadata_rows(ex)); prediction['label']=labels(ex); prediction['probability']=p
        prediction.to_csv(out/'predictions'/f'{stem}_{split}.csv',index=False)
    if model in MODELS[:3]:joblib.dump({'model':joblib.load(path),**protocol,'threshold':thresholds[90]},path)
    (out/'arm_metrics'/f'{stem}.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    return stem,fit

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    ap.add_argument('--horizons',nargs='+',type=int,choices=[6,12,24],default=[6,12,24])
    ap.add_argument('--variants',nargs='+',choices=['structured','partial_ontology'],default=['structured','partial_ontology'])
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--output-dir',type=Path,default=None)
    ap.add_argument('--protocol-version',type=int,choices=[4,5],default=5)
    args=ap.parse_args(); out=args.output_dir or OUT.with_name(f'paired_partial_ontology_quality_corrected_v{args.protocol_version}')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.training.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another training process owns this output directory')
    for name in ['model_artifacts','predictions','arm_metrics','logs','prepared_bundles']:(out/name).mkdir(parents=True,exist_ok=True)
    from clinical_feature_contract import EXCLUDED_FEATURES,EXCLUDED_FEATURES_V5
    if args.protocol_version==5:EXCLUDED_FEATURES=EXCLUDED_FEATURES_V5
    config={**{k:v for k,v in vars(args).items() if k!='protocol_version'},'output_dir':str(out),'preprocessing_version':args.protocol_version,'recurrent_pooling':'final_hidden',
        'excluded_features':sorted(EXCLUDED_FEATURES),'numeric_contract':'strict assay names and explicit units; all raw events with per-assay as-of',
        'seed_policy':'base_seed + horizon; reset independently before each model/arm',
        'selection':'local validation AUPRC only; no external selection'}
    config_path=out/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Existing output directory has a different protocol; choose a new directory')
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2))
    jobs=[]
    for h in args.horizons:
        task=f'pre_{h}h'; raw=build_bundle(task,'structured',DATA,ONTO,SCHEMA,CACHE,24,
            numeric_contract=f'v{args.protocol_version}',audit_path=out/f'{task}_source_repair_audit.json')
        partial=make_partial_bundle(raw)
        indices=[partial.feature_names.index(n) for n in raw.feature_names]
        for a,b in zip(raw.examples,partial.examples):
            assert (a.patient_id,a.visit_no,a.endpoint_time,a.split,a.label)==(b.patient_id,b.visit_no,b.endpoint_time,b.split,b.label)
            assert np.array_equal(a.x,b.x[:,indices])
        patient_sets={s:{e.patient_id for e in raw.by_split(s)} for s in ['train','validation','test']}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            if patient_sets[a]&patient_sets[b]:raise ValueError('Patient leakage across local splits')
        for variant in args.variants:
            bundle=clone(raw if variant=='structured' else partial,variant)
            scaler=FeatureScaler(bundle.feature_names).fit(bundle.by_split('train')); scaler.transform(bundle.examples)
            summary={'task':task,'variant':variant,'feature_names':bundle.feature_names,'scaler':scaler.state_dict(),
                'counts':{s:{'n':len(bundle.by_split(s)),'positive':int(labels(bundle.by_split(s)).sum())} for s in patient_sets}}
            (out/f'{task}_{variant}_dataset_summary.json').write_text(json.dumps(summary,indent=2))
            bundle_path=out/'prepared_bundles'/f'{task}_{variant}.pkl'
            with bundle_path.open('wb') as f:pickle.dump((bundle,scaler.state_dict()),f)
            for model in args.models:
                if not (out/'arm_metrics'/f'{task}_{variant}_{model}.json').exists():
                    jobs.append((str(bundle_path),model,str(out),args.seed+h,args.epochs,args.protocol_version))
        print('Prepared',task,flush=True)
    pool=ProcessPoolExecutor(max_workers=args.workers)
    try:
        for future in as_completed([pool.submit(train_job,j) for j in jobs]):
            stem,fit=future.result(); print('Completed',stem,fit,flush=True)
    except BaseException:
        for process in pool._processes.values():process.terminate()
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    else:pool.shutdown(wait=True)
    records=[]
    for path in sorted((out/'arm_metrics').glob('*.json')):records.extend(json.loads(path.read_text()))
    with (out/'metrics.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print('Completed',len(records),'metric records',flush=True)

if __name__=='__main__':main()
