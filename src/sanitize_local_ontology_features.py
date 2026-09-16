"""Create a quality-controlled copy of local ontology JSONL model inputs.

Fixes known name/unit mapping gaps and removes clearly non-physiologic values
before fitting scalers.  The original ontology_enhanced directory is untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


RANGES = {
    "value__vital__temperature": (25.0, 45.0),
    "value__vital__heart_rate": (20.0, 250.0),
    "value__vital__respiratory_rate": (3.0, 100.0),
    "value__vital__oxygen_saturation": (50.0, 100.0),
    "value__vital__systolic_bp": (40.0, 300.0),
    "value__vital__diastolic_bp": (20.0, 200.0),
    "value__vital__mean_bp": (20.0, 250.0),
    "value__lab__wbc": (0.1, 500.0),
    "value__lab__neutrophil": (0.1, 100.0),
    "value__lab__platelet": (1.0, 2000.0),
    "value__lab__hemoglobin": (20.0, 250.0),
    "value__lab__creatinine": (1.0, 5000.0),
    "value__lab__bilirubin": (0.1, 1000.0),
    "value__lab__lactate": (0.1, 30.0),
    "value__lab__pao2": (5.0, 700.0),
    "value__lab__ph": (6.5, 8.0),
    "value__lab__sodium": (100.0, 200.0),
    "value__lab__potassium": (1.0, 10.0),
    "value__lab__chloride": (50.0, 180.0),
}

INVALIDATES = {
    "value__vital__temperature": {"Fever", "Hypothermia"},
    "value__vital__heart_rate": {"Tachycardia", "Bradycardia"},
    "value__vital__respiratory_rate": {"Tachypnea"},
    "value__vital__oxygen_saturation": {"Hypoxemia", "RespiratoryDysfunction"},
    "value__vital__systolic_bp": {"Hypotension", "CardiovascularDysfunction"},
    "value__vital__mean_bp": {"Hypotension", "CardiovascularDysfunction"},
    "value__lab__wbc": {"Leukocytosis", "Leukopenia", "InflammatoryResponse", "SystemicInflammatoryResponse"},
    "value__lab__neutrophil": {"Neutrophilia"},
    "value__lab__platelet": {"Thrombocytopenia", "CoagulationDysfunction"},
    "value__lab__hemoglobin": {"Anemia"},
    "value__lab__creatinine": {"RenalDysfunction"},
    "value__lab__bilirubin": {"HepaticDysfunction"},
    "value__lab__lactate": {"ElevatedLactate", "MetabolicDysfunction"},
    "value__lab__pao2": {"Hypoxemia", "RespiratoryDysfunction"},
    "value__lab__ph": {"MetabolicAcidosis", "MetabolicDysfunction"},
    "value__lab__sodium": {"Hypernatremia"},
    "value__lab__potassium": {"Hypokalemia"},
    "value__lab__chloride": {"Hyperchloremia"},
}


def finite_number(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def is_neutrophil_percent(item: dict) -> bool:
    text = f"{item.get('source_text') or ''} {item.get('raw_value') or ''}"
    name = text.replace(" ", "")
    return ("中性分叶核粒细胞百分率" in name or "中性粒细胞百分比" in name
            or "中性粒细胞百分率" in name)


def sanitise_record(record: dict) -> tuple[dict, int, int]:
    mf = record.get("model_features") or {}
    order = list(mf.get("feature_order") or [])
    vec = list(mf.get("feature_vector") or [])
    if not order or len(order) != len(vec):
        return record, 0, 0
    idx = {name: i for i, name in enumerate(order)}
    masks = {name: i for i, name in enumerate(order) if name.startswith("mask__value__")}
    invalid_count = 0
    neut_count = 0

    # Recover the omitted local neutrophil percentage mapping. Absolute counts
    # are deliberately excluded because the model definition uses percentages.
    for item in record.get("structured_observations") or []:
        if is_neutrophil_percent(item):
            item["feature_id"] = "value__lab__neutrophil"
            item["feature_concept"] = "NeutrophilMeasurement"
            item["feature_eligible"] = True
            x = finite_number(item.get("value"))
            if x is not None and "value__lab__neutrophil" in idx:
                vec[idx["value__lab__neutrophil"]] = x
                if "mask__value__lab__neutrophil" in idx:
                    vec[idx["mask__value__lab__neutrophil"]] = 1.0
                neut_count += 1

    for feature, (lo, hi) in RANGES.items():
        j = idx.get(feature)
        if j is None:
            continue
        x = finite_number(vec[j])
        if x is None or (x != 0 and not (lo <= x <= hi)):
            if x not in (None, 0.0):
                invalid_count += 1
            vec[j] = 0.0
            mask_j = idx.get("mask__" + feature)
            if mask_j is not None:
                vec[mask_j] = 0.0
            for state in INVALIDATES.get(feature, ()):
                sj = idx.get("state__" + state)
                if sj is not None:
                    vec[sj] = 0.0

    # Recompute the corrected neutrophilia state from the recovered percentage.
    nj = idx.get("value__lab__neutrophil")
    sj = idx.get("state__Neutrophilia")
    if nj is not None and sj is not None:
        x = finite_number(vec[nj])
        vec[sj] = 1.0 if x is not None and x > 75 else 0.0

    mf["feature_vector"] = vec
    values = {name: vec[i] for i, name in enumerate(order) if finite_number(vec[i]) not in (None, 0.0)}
    mf["feature_values"] = values
    observed = [name for name, i in masks.items() if finite_number(vec[i]) not in (None, 0.0)]
    mf["observed_features"] = sorted(set(observed))
    record["model_features"] = mf
    return record, invalid_count, neut_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("ProcessedData/model_dataset/ontology_enhanced"))
    ap.add_argument("--output-dir", type=Path, default=Path("ProcessedData/model_dataset/ontology_enhanced_quality_cleaned"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_invalid = total_neut = total_records = 0
    for src in sorted(args.input_dir.iterdir()):
        dst = args.output_dir / src.name
        if not src.name.endswith(".jsonl"):
            if src.is_file(): shutil.copy2(src, dst)
            continue
        with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip(): continue
                rec, bad, neut = sanitise_record(json.loads(line))
                fout.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                total_records += 1; total_invalid += bad; total_neut += neut
        print(src.name, "done")
    print(json.dumps({"records": total_records, "invalid_values_removed": total_invalid, "neutrophil_rows_recovered": total_neut}, ensure_ascii=False))


if __name__ == "__main__": main()
