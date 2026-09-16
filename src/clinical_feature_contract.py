"""Version 4 assay contract, defined from names/units before external scoring.

Rebuild local laboratory channels from source snapshots, never cached LLM
numbers. Missing units are not inferred from a numerical distribution.
"""
import math
import re
from collections import Counter, defaultdict

VERSION = 4
# No unit/method metadata in the external D-dimer export; local RR-Set is a
# ventilator setting, whereas the external column is measured respiratory rate.
# Exclude both value and mask in BOTH arms and BOTH datasets until re-extracted.
EXCLUDED_FEATURES = {'value__lab__d_dimer', 'value__vital__respiratory_rate'}
EXCLUDED_FEATURES_V5 = EXCLUDED_FEATURES | {'value__vital__glucose'}
ARTERIAL_FEATURES = {'value__lab__'+n for n in ('ph','pao2','paco2','hco3','base_excess','oxygenation_index','oxygen_saturation')}

PATTERNS = {
    'lactate': r'^(?:全血|血浆|血清)?乳酸$',
    'procalcitonin': r'^(?:降钙素原|PCT)$',
    'crp': r'^(?:超敏)?C-?反应蛋白$|^CRP$',
    'wbc': r'^(?:白细胞计数|白血球计数|WBC)$',
    'neutrophil': r'^中性(?:分叶核)?粒细胞(?:百分率|百分比)$',
    'platelet': r'^(?:阻抗法)?血小板(?:计数)?$',
    'hemoglobin': r'^(?:血红蛋白(?:总浓度)?|血色素|HGB|Hb)$',
    'creatinine': r'^(?:血清)?肌酐$',
    'urea': r'^(?:血清)?尿素(?:氮)?$',
    'bilirubin': r'^(?:血清)?总胆红素$',
    'albumin': r'^(?:血|血清)?白蛋白$',
    'glucose': r'^(?:全血|血浆|血清)?(?:葡萄糖|血糖)$|^餐后2小时血糖$|^BS$',
    'ph': r'^(?:酸碱度|PH)$',
    'pao2': r'^氧分压$',
    'paco2': r'^二氧化碳分压$',
    'hco3': r'^碳酸氢根(?:浓度)?$',
    'base_excess': r'^全血碱剩余$',
    'oxygenation_index': r'^氧合指数$',
    'oxygen_saturation': r'^氧饱和度$',
    'sodium': r'^(?:全血|血清)?钠$',
    'potassium': r'^(?:全血|血清)?钾$',
    'chloride': r'^(?:全血|血清)?氯$',
    # The external chemistry field is TOTAL calcium, not ionized calcium.
    'calcium': r'^(?:血清|总|血清总)?钙$',
    'pt': r'^(?:凝血酶原时间|PT)$',
    'inr': r'^(?:国际标准化比值|INR)$',
    'aptt': r'^(?:活化部分凝血活酶时间|APTT)$',
    'fibrinogen': r'^(?:纤维蛋白原(?:演算值)?|FIB)$',
    'd_dimer': r'^D-?二聚体$',
}
COMPILED = [(f'value__lab__{k}', re.compile(v, re.I)) for k,v in PATTERNS.items()]
NON_BLOOD_PANEL = re.compile(r'尿液|尿常规|尿沉渣|尿电解质|24小时尿|粪便|大便|脑脊液|体液|胸水|腹水|透析液')

def normalise_name(name):
    return re.sub(r'^(?:\s|[☆★*#]|\[HR\])+', '', str(name or ''), flags=re.I).strip()

def assay_feature(item):
    if NON_BLOOD_PANEL.search(str(item.get('code') or '')):
        return None
    name = normalise_name(item.get('name'))
    if NON_BLOOD_PANEL.search(name):
        return None
    return next((feature for feature,pattern in COMPILED if pattern.fullmatch(name)), None)

def parse_exact_number(value):
    """Do not turn '<0.1', '1:900' or a textual range into exact observations."""
    text = str(value if value is not None else '').strip().replace('−','-')
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', text):
        return None
    number = float(text)
    return number if math.isfinite(number) else None

def normalise_unit(unit):
    return str(unit or '').lower().replace('μ','u').replace('µ','u').replace(' ','').replace('／','/')

def canonical_unit(feature):
    key=feature.removeprefix('value__lab__')
    if key in ('wbc','platelet'):return '10^9/L'
    if key in ('neutrophil','oxygen_saturation'):return '%'
    if key in ('hemoglobin','albumin','fibrinogen'):return 'g/L'
    if key in ('creatinine','bilirubin'):return 'umol/L'
    if key in ('pao2','paco2','oxygenation_index'):return 'mmHg'
    if key in ('pt','aptt'):return 's'
    if key in ('ph','inr'):return '1'
    if key=='crp':return 'mg/L'
    if key=='procalcitonin':return 'ng/mL'
    if key=='d_dimer':return 'mg/L FEU'
    return 'mmol/L'

def lab_value(item):
    feature = assay_feature(item)
    if feature is None: return None, 'other_assay'
    raw = item.get('value')
    if raw in (None, ''): raw = item.get('result')
    value = parse_exact_number(raw)
    if value is None: return None, 'non_exact_numeric'
    key=feature.removeprefix('value__lab__'); unit=normalise_unit(item.get('unit'))
    units={'mmol/l':1.}
    if key in ('wbc','platelet'): units={'10^9/l':1.,'10*9/l':1.}
    elif key in ('neutrophil','oxygen_saturation'): units={'%':1.}
    elif key in ('hemoglobin','albumin'): units={'g/l':1.,'g/dl':10.}
    elif key=='creatinine': units={'umol/l':1.,'mmol/l':1000.,'mg/dl':88.4}
    elif key=='bilirubin': units={'umol/l':1.,'mg/dl':17.104}
    elif key=='glucose': units={'mmol/l':1.,'mg/dl':1/18.018}
    elif key=='urea':
        units={'mmol/l':1., 'mg/dl': .357 if '氮' in normalise_name(item.get('name')) else 1/6.006}
    elif key=='calcium': units={'mmol/l':1.,'mg/dl':1/4.008}
    elif key in ('pao2','paco2','oxygenation_index'): units={'mmhg':1.,'kpa':7.50062}
    elif key in ('pt','aptt'): units={'秒':1.,'s':1.,'sec':1.}
    elif key in ('ph','inr'): units={'':1.,'1':1.}
    elif key=='crp': units={'mg/l':1.,'mg/dl':10.}
    elif key=='procalcitonin': units={'ng/ml':1.,'ug/l':1.}
    elif key=='fibrinogen': units={'g/l':1.,'mg/dl':.01,'mg/l':.001}
    elif key=='d_dimer': units={'mg/lfeu':1.}  # audited locally, excluded from transfer
    if unit not in units: return None,'unknown_or_incompatible_unit'
    value *= units[unit]
    from sanitize_local_ontology_features import RANGES
    limits=RANGES.get(feature)
    if limits and not limits[0] <= value <= limits[1]: return None,'outside_existing_quality_range'
    if key=='glucose' and not 0 < value <= 10000/18.018: return None,'outside_existing_quality_range'
    if key not in ('base_excess',) and value < 0: return None,'negative_concentration'
    return (feature,value),'accepted'

def rebuild_labs(rows, table, names, key_fn, time_fn):
    idx={n:i for i,n in enumerate(names)}
    features=[n for n in names if n.startswith('value__lab__')]
    counts=Counter(); changes=Counter(); previous_visit=None; carried={}; availability={}
    for row in sorted(rows,key=lambda r:(str(r.get('split','')),str(r.get('visit_no','')),time_fn(r.get('round_time')))):
        visit=(str(row.get('split','')),str(row.get('visit_no','')))
        if visit!=previous_visit: carried={}; previous_visit=visit
        key=key_fn(row)
        if key not in table: raise ValueError('Source row missing from cached feature keys')
        cutoff=time_fn(row.get('round_time')); candidates=defaultdict(list)
        for item in row.get('latest_lab') or []:
            found,reason=lab_value(item); counts[reason]+=1
            if found is None: continue
            feature,value=found; t=time_fn(item.get('latest_time'))
            if not t or t>cutoff: counts['future_or_missing_time']+=1; continue
            candidates[feature].append((t,value))
        vector=table[key].copy()
        for feature in features:
            new=False
            options=candidates.get(feature,[])
            if options:
                latest=max(t for t,v in options); values={v for t,v in options if t==latest}
                if len(values)>1: counts['same_time_conflicting_values']+=1
                elif feature not in carried or latest>carried[feature][0]:
                    carried[feature]=(latest,next(iter(values))); new=True
            value=carried.get(feature,('',0.))[1]
            if vector[idx[feature]]!=value: changes[feature]+=1
            vector[idx[feature]]=value
            vector[idx['mask__'+feature]]=float(new)
        table[key]=vector
        availability[key]=set(carried)
    return {'snapshot_item_counts':dict(counts),'changed_value_rows':dict(changes),'source_rows':len(rows)},availability

def rebuild_from_all_events(rows, table, names, key_fn, time_fn, event_path, include_vitals=False):
    """As-of carry for EACH assay, using all lab events rather than last panel."""
    from datetime import timedelta
    import polars as pl
    idx={n:i for i,n in enumerate(names)}; features=[n for n in names if n.startswith('value__lab__') or (include_vitals and n.startswith('value__vital__'))]
    by_visit=defaultdict(list)
    for row in rows: by_visit[str(row['visit_no'])].append(row)
    source=pl.read_parquet(event_path).filter(pl.col('visit_no').is_in(list(by_visit)))
    if source.is_empty():raise ValueError('No canonical source labs for selected visits')
    events=defaultdict(list)
    for visit,t,feature,value in source.select('visit_no','event_time','feature','value').iter_rows():
        events[visit].append((t,feature,value))
    available={};changes=Counter();recovered=Counter();counts=Counter()
    for visit,visit_rows in by_visit.items():
        visit_rows.sort(key=lambda r:time_fn(r['round_time']))
        admission=visit_rows[0]['round_time']-timedelta(hours=float(visit_rows[0]['hours_from_admission']))
        ev=sorted((t,f,v) for t,f,v in events[visit] if t>=admission)
        pointer=0;carried={}
        for row in visit_rows:
            cutoff=row['round_time'];key=key_fn(row);new=set()
            while pointer<len(ev) and ev[pointer][0]<cutoff:
                t,f,v=ev[pointer];carried[f]=v;new.add(f);pointer+=1
            vector=table[key].copy()
            for f in features:
                value=carried.get(f,0.)
                if f in carried and vector[idx[f]]==0 and vector[idx['mask__'+f]]==0:recovered[f]+=1
                if vector[idx[f]]!=value:changes[f]+=1
                vector[idx[f]]=value;vector[idx['mask__'+f]]=float(f in new)
            table[key]=vector;available[key]=set(carried)
            counts['source_rows']+=1;counts['available_lab_cells']+=len(carried)
        counts['canonical_events_consumed']+=pointer
    return {**dict(counts),'changed_value_rows':dict(changes),'available_where_old_cache_missing':dict(recovered),
        'source':'all raw laboratory events; per-assay causal as-of; max(lab_time, check_time)'},available
