"""从本体和真实时间点数据生成固定的模型特征目录。

结构化数据只保留脓毒症早期预测相关的规范化指标；查房文本只保留
WARD_RULES 实际在数据中命中的本体概念；关系只保留有临床含义的关系计数。

输出：ProcessedData/feature_schema.json

示例：
    python build_feature_schema.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

from map_one_case_with_llm import (
    AGE_FEATURE_ID,
    AGE_RISK_THRESHOLD,
    CHRONIC_DIAGNOSIS_CONCEPTS,
    DIAGNOSIS_FILE,
    DIAGNOSIS_RULES,
    MEASUREMENT_CONCEPTS,
    NON_FEATURE_CONCEPTS,
    OUTCOME_WORDS,
    REQUIRED_CONTEXT_CONCEPTS,
    REQUIRED_WARD_CONCEPTS,
    RELATION_FEATURES,
    WARD_RULES,
    age_at,
    numeric_value,
    structured_feature,
)


DATA_FILE = Path("ProcessedData/model_dataset/timepoint_dataset.parquet")
ONTOLOGY_FILE = Path("ontology/sepsis_prediction_ontology.json")
BASE_FILE = Path("Dataset/baseinfo.csv")
OUTPUT_FILE = Path("ProcessedData/feature_schema.json")


def registry(ontology):
    return {
        concept: branch
        for branch, concepts in ontology["classes"].items()
        for concept in concepts
    }


def usable_numeric(item):
    """value/result 二选一；只有能解析成数字的项目才进入数值矩阵。"""
    raw = item.get("value")
    if raw in (None, ""):
        raw = item.get("result")
    return numeric_value(raw) is not None


def relation_definition(predicate, ontology):
    domain, range_ = ontology["object_properties"].get(
        predicate, ["ClinicalObservation", "owl:Thing"]
    )
    return {
        "predicate": predicate,
        "domain": domain,
        "range": range_,
        "aggregation": "count_per_timepoint",
    }


def demographic_stats(data):
    """统计 baseinfo 可覆盖的时间点和年龄风险时间点。"""
    if not BASE_FILE.exists():
        return 0, 0
    base = pl.read_csv(BASE_FILE, encoding="utf8-lossy").select([
        pl.col("就诊号").cast(pl.String).alias("visit_no"),
        pl.col("出生日期"),
    ])
    birth_by_visit = {
        row["visit_no"]: row["出生日期"]
        for row in base.to_dicts()
    }
    numeric_count = 0
    age_risk_count = 0
    for row in data.select(["visit_no", "round_time"]).iter_rows(named=True):
        birth_date = birth_by_visit.get(row["visit_no"])
        if not birth_date:
            continue
        age = age_at(birth_date, row["round_time"])
        if age is None:
            continue
        numeric_count += 1
        age_risk_count += int(age >= AGE_RISK_THRESHOLD)
    return numeric_count, age_risk_count


def diagnosis_counts(concepts):
    """只统计 cutoff 前可用于建模的诊断概念覆盖情况。"""
    counts = Counter()
    if not DIAGNOSIS_FILE.exists():
        return counts
    diagnosis = pl.read_csv(
        DIAGNOSIS_FILE,
        encoding="utf8-lossy",
        infer_schema_length=1000,
    )
    for row in diagnosis.to_dicts():
        text = str(row.get("visit_diag_diagicddesc") or "")
        if (
            not text
            or row.get("visit_diag_diagtypename") == "出院诊断"
            or re.search(OUTCOME_WORDS, text, flags=re.I)
        ):
            continue
        matched = []
        for pattern, concept in DIAGNOSIS_RULES:
            if concept in concepts and re.search(pattern, text, flags=re.I):
                matched.append(concept)
        for concept in set(matched):
            counts[concept] += 1
        if any(concept in CHRONIC_DIAGNOSIS_CONCEPTS for concept in matched):
            counts["ChronicDisease"] += 1
        if any(concept in {
            "BloodstreamInfection", "CentralNervousSystemInfection",
            "SkinAndSoftTissueInfection", "PostoperativeInfection",
            "InfectionSource", "PositiveCulture", "NegativeCulture",
        } for concept in matched):
            counts["InfectionEvidence"] += 1
    return counts


def main():
    ontology = json.loads(ONTOLOGY_FILE.read_text(encoding="utf-8"))
    concepts = registry(ontology)
    data = pl.read_parquet(DATA_FILE)
    age_count, age_risk_count = demographic_stats(data)

    # 一个规范化 feature_id 对应多个医院原始名称，例如“胆红素”和“总胆红素”。
    structured = defaultdict(lambda: {
        "source_types": set(),
        "source_names": set(),
        "ontology_concepts": set(),
        "observed_count": 0,
        "numeric_count": 0,
    })
    text_counts = Counter()

    for row in data.iter_rows(named=True):
        for source_type, items in (
            ("vital", row["latest_vital"]),
            ("lab", row["latest_lab"]),
        ):
            for item in items or []:
                feature_id, concept = structured_feature(item, source_type, concepts)
                if not feature_id or concept in NON_FEATURE_CONCEPTS:
                    continue
                info = structured[feature_id]
                info["source_types"].add(source_type)
                info["source_names"].add(item.get("name") or "")
                info["ontology_concepts"].add(concept)
                info["observed_count"] += 1
                info["numeric_count"] += int(usable_numeric(item))

        text = row.get("round_text") or ""
        for pattern, concept in WARD_RULES:
            if (
                concept in concepts
                and concept not in NON_FEATURE_CONCEPTS
                and re.search(pattern, text, flags=re.I)
            ):
                text_counts[concept] += 1

    text_counts.update(diagnosis_counts(concepts))
    text_counts["AgeRelatedRisk"] += age_risk_count

    # 没有任何可解析数值的规范化指标不进入矩阵，但保留在审计信息中。
    numeric_catalog = {
        feature_id: info
        for feature_id, info in structured.items()
        if info["numeric_count"] > 0
    }

    features = []
    structured_feature_catalog = []
    for feature_id in sorted(numeric_catalog):
        info = numeric_catalog[feature_id]
        metadata = {
            "ontology_concepts": sorted(info["ontology_concepts"]),
            "source_types": sorted(info["source_types"]),
            "source_names": sorted(name for name in info["source_names"] if name),
            "observed_count": info["observed_count"],
            "numeric_count": info["numeric_count"],
        }
        features.append({
            "feature_id": feature_id,
            "kind": "numeric",
            "dtype": "float32",
            "default": 0.0,
            "carry_forward": True,
            **metadata,
        })
        features.append({
            "feature_id": f"mask__{feature_id}",
            "kind": "numeric_mask",
            "dtype": "float32",
            "default": 0.0,
            "value_feature": feature_id,
            "mask_semantics": "1=current_timepoint_observed; 0=carried_forward_or_missing",
            **metadata,
        })
        structured_feature_catalog.append({
            "feature_id": feature_id,
            "ontology_concepts": sorted(info["ontology_concepts"]),
            "source_types": sorted(info["source_types"]),
            "source_names": sorted(name for name in info["source_names"] if name),
            "observed_count": info["observed_count"],
            "numeric_count": info["numeric_count"],
        })

    # 年龄不是检验/体征，但它是 baseinfo 中可重复计算的基线数值特征。
    if age_count:
        age_metadata = {
            "ontology_concepts": ["AgeRelatedRisk"],
            "source_types": ["baseinfo"],
            "source_names": ["出生日期"],
            "observed_count": age_count,
            "numeric_count": age_count,
        }
        features.extend([
            {
                "feature_id": AGE_FEATURE_ID,
                "kind": "numeric",
                "dtype": "float32",
                "default": 0.0,
                "carry_forward": True,
                **age_metadata,
            },
            {
                "feature_id": f"mask__{AGE_FEATURE_ID}",
                "kind": "numeric_mask",
                "dtype": "float32",
                "default": 0.0,
                "value_feature": AGE_FEATURE_ID,
                "mask_semantics": "1=current_timepoint_observed; 0=carried_forward_or_missing",
                **age_metadata,
            },
        ])
        structured_feature_catalog.append({
            "feature_id": AGE_FEATURE_ID,
            **age_metadata,
        })

    supported_concepts = set()
    for info in numeric_catalog.values():
        supported_concepts.update(info["ontology_concepts"])
    supported_concepts.update(text_counts)
    supported_concepts.update(REQUIRED_CONTEXT_CONCEPTS)
    supported_concepts -= NON_FEATURE_CONCEPTS

    for concept in sorted(supported_concepts):
        if concept in MEASUREMENT_CONCEPTS:
            continue
        features.append({
            "feature_id": f"state__{concept}",
            "kind": "ontology_state",
            "dtype": "float32",
            "default": 0.0,
            "ontology_concept": concept,
            "ontology_branch": concepts[concept],
            "state_codes": {
                "0": "not_mentioned",
                "1": "confirmed_present",
                "2": "uncertain",
                "3": "explicitly_negated",
            },
            "observed_count": text_counts[concept] + sum(
                info["numeric_count"]
                for info in numeric_catalog.values()
                if concept in info["ontology_concepts"]
            ),
            "support_status": (
                "observed_in_training_data"
                if text_counts[concept] > 0
                else "configured_for_future_observation"
            ),
        })

    relation_features = []
    for predicate in RELATION_FEATURES:
        feature_id = f"relation_count__{predicate}"
        features.append({
            "feature_id": feature_id,
            "kind": "relation_count",
            "dtype": "float32",
            "default": 0.0,
            **relation_definition(predicate, ontology),
        })
        relation_features.append(feature_id)

    feature_groups = {
        "structured_numeric": [
            x["feature_id"] for x in features if x["kind"] == "numeric"
        ],
        "structured_numeric_mask": [
            x["feature_id"] for x in features if x["kind"] == "numeric_mask"
        ],
        "ontology_state": [
            x["feature_id"] for x in features if x["kind"] == "ontology_state"
        ],
        "ontology_relation": relation_features,
    }

    schema = {
        "schema_id": "SEPPRED_FEATURES",
        "schema_version": "0.7.0",
        "ontology_id": ontology["ontology_id"],
        "ontology_version": ontology["version"],
        "data_file": str(DATA_FILE),
        "data_sources": [
            str(DATA_FILE),
            str(BASE_FILE),
            str(DIAGNOSIS_FILE),
        ],
        "feature_dim": len(features),
        "feature_order": [item["feature_id"] for item in features],
        "feature_groups": feature_groups,
        "features": features,
        "structured_feature_catalog": structured_feature_catalog,
        "data_supported_concepts": sorted(supported_concepts),
        "unsupported_ontology_concepts": sorted(
            set(concepts) - supported_concepts - NON_FEATURE_CONCEPTS
        ),
        "relation_definitions": [
            relation_definition(predicate, ontology)
            for predicate in RELATION_FEATURES
        ],
        "notes": [
            "Only curated canonical numeric features are included; raw hospital item names are aliases, not dimensions.",
            "Lab/vital values are read from value first and result second.",
            "Later timepoints may carry forward the latest earlier numeric value; mask=1 means newly observed and mask=0 means carried-forward or missing.",
            "Ontology state features use 0=not mentioned, 1=confirmed present, 2=uncertain, 3=explicitly negated.",
            "Relation features aggregate clinical Subject-Predicate-Object edges at the current timepoint.",
            "Structural graph relations are retained as graph_relations for audit only and are excluded from model relation counts.",
            "Sepsis outcome concepts and post-cutoff diagnosis information are excluded from predictor features.",
            "Selected ward-round concepts are kept as fixed columns even when their current training count is zero, so split schemas remain identical.",
            "AgeRelatedRisk is derived from baseinfo birth date with a 65-year threshold; age itself is retained as value__demographic__age.",
            "Pre-cutoff diagnosis concepts are mapped from icu_diagnosis_HID0101.csv; discharge diagnoses and sepsis outcome mentions are excluded.",
        ],
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"feature_dim: {schema['feature_dim']}")
    print(f"structured numeric features: {len(feature_groups['structured_numeric'])}")
    print(f"supported ontology concepts: {len(supported_concepts)}")
    print(f"relation features: {len(relation_features)}")
    print(f"output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
