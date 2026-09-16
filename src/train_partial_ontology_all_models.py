#!/usr/bin/env python3
"""Compatibility entry point for matched structured/partial-ontology training."""
from __future__ import annotations
from pathlib import Path
from temporal_data import SequenceExample, SequenceBundle

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ProcessedData/model_dataset/pre_sepsis_anchors"
ONTO = ROOT / "ProcessedData/model_dataset/ontology_enhanced_quality_cleaned"
CACHE = ROOT / "ProcessedData/model_dataset/temporal_cache_quality_cleaned"
SCHEMA = ROOT / "ProcessedData/feature_schema.json"
MODELS = ("logreg", "random_forest", "xgboost", "lstm", "gru", "gru_d", "transformer")


def clone(bundle, variant):
    return SequenceBundle(bundle.task, variant, list(bundle.feature_names), [SequenceExample(
        x=e.x.copy(), delta_hours=e.delta_hours.copy(), hours_from_admission=e.hours_from_admission.copy(),
        observed_mask=e.observed_mask.copy(), label=e.label, split=e.split, patient_id=e.patient_id,
        visit_no=e.visit_no, endpoint_time=e.endpoint_time,
        available_mask=e.available_mask.copy() if getattr(e,'available_mask',None) is not None else None) for e in bundle.examples], bundle.endpoints)


def main():
    """Keep the original entry point, using versioned matched training."""
    from train_matched_models import main as matched_main
    matched_main()


if __name__ == "__main__":
    main()
