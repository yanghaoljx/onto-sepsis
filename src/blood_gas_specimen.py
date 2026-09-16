"""Match exported blood gas rows to source specimen metadata, failing closed."""
from datetime import datetime
from collections import defaultdict
import csv
from clinical_feature_contract import ARTERIAL_FEATURES

def load_specimen_lookup(path):
    groups=defaultdict(set)
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not {'subject_id','hadm_id','charttime','specimen'} <= set(reader.fieldnames or []):
            raise ValueError('Blood gas lookup needs subject_id, hadm_id, charttime, specimen columns')
        for row in reader:
            if not row['hadm_id']:continue
            key=(int(row['subject_id']),int(row['hadm_id']),datetime.fromisoformat(row['charttime']))
            groups[key].add(row['specimen'].strip().upper())
    if not groups:raise ValueError('Empty blood gas specimen lookup')
    # Mixed specimen types at the same timestamp cannot be disambiguated in
    # the old export; retain arterial fields only for unambiguously ART rows.
    return {k:types=={'ART.'} for k,types in groups.items()}

def allowed_blood_gas_feature(feature,is_arterial):
    return is_arterial or feature not in ARTERIAL_FEATURES
