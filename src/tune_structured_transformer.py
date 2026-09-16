#!/usr/bin/env python3
"""Validation-only tuning for the structured-input time-aware Transformer.

The ontology-selected architecture must not be reused as if it had also been
selected for the lower-dimensional structured representation.  This script
uses a small, prespecified validation grid and evaluates the independent test
set only once after the winning configuration is locked.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from temporal_data import FeatureScaler, build_bundle, labels
from tune_time_aware_transformer import fit_config, predict


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ProcessedData/temporal_results/structured_transformer_validation_tuning_carryforward_median"
DATA_DIR = ROOT / "ProcessedData/model_dataset/pre_sepsis_anchors"
ONTOLOGY_DIR = ROOT / "ProcessedData/model_dataset/ontology_enhanced"
CACHE_DIR = ROOT / "ProcessedData/model_dataset/temporal_cache"
BASE_RESULTS = ROOT / "ProcessedData/temporal_results/main"
SEED = 20260901

# Prespecified before test evaluation. The grid is deliberately compact to
# limit validation-selection optimism in a cohort of roughly 5,500 training
# admissions per horizon.
CONFIGS = [
    {
        "name": "legacy_last_128",
        "architecture": "legacy",
        "hidden": 128,
        "heads": 4,
        "layers": 3,
        "dropout": 0.15,
        "lr": 2e-4,
        "wd": 1e-4,
        "pooling": "last",
        "batch_size": 128,
        "max_epochs": 30,
        "patience": 6,
        "min_epochs": 10,
        "lr_scheduler": True,
    },
    {
        "name": "attention_96",
        "architecture": "enhanced",
        "hidden": 96,
        "heads": 4,
        "layers": 2,
        "dropout": 0.15,
        "lr": 3e-4,
        "wd": 1e-4,
        "pooling": "attention",
        "batch_size": 128,
        "max_epochs": 30,
        "patience": 6,
        "min_epochs": 10,
        "lr_scheduler": True,
    },
    {
        "name": "attention_128",
        "architecture": "enhanced",
        "hidden": 128,
        "heads": 4,
        "layers": 2,
        "dropout": 0.10,
        "lr": 3e-4,
        "wd": 1e-5,
        "pooling": "attention",
        "batch_size": 128,
        "max_epochs": 30,
        "patience": 6,
        "min_epochs": 10,
        "lr_scheduler": True,
    },
    {
        "name": "multiscale_96",
        "architecture": "enhanced",
        "hidden": 96,
        "heads": 4,
        "layers": 2,
        "dropout": 0.15,
        "lr": 3e-4,
        "wd": 1e-4,
        "pooling": "multiscale",
        "batch_size": 128,
        "max_epochs": 30,
        "patience": 6,
        "min_epochs": 10,
        "lr_scheduler": True,
    },
    {
        "name": "multiscale_128",
        "architecture": "enhanced",
        "hidden": 128,
        "heads": 4,
        "layers": 2,
        "dropout": 0.20,
        "lr": 2e-4,
        "wd": 1e-4,
        "pooling": "multiscale",
        "batch_size": 128,
        "max_epochs": 30,
        "patience": 6,
        "min_epochs": 10,
        "lr_scheduler": True,
    },
]


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(y, probability)),
        "auroc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def paired_delta_ci(
    y: np.ndarray, new: np.ndarray, reference: np.ndarray, seed: int, reps: int = 3000
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    draws = []
    point = average_precision_score(y, new) - average_precision_score(y, reference)
    for _ in range(reps):
        index = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        draws.append(
            average_precision_score(y[index], new[index])
            - average_precision_score(y[index], reference[index])
        )
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_auprc_vs_xgboost": float(point),
        "delta_auprc_vs_xgboost_lower": float(low),
        "delta_auprc_vs_xgboost_upper": float(high),
        "bootstrap_n": reps,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    leaderboard = []
    selected_rows = []
    for horizon_index, horizon in enumerate((6, 12, 24)):
        task = f"pre_{horizon}h"
        bundle = build_bundle(
            task,
            "structured",
            DATA_DIR,
            ONTOLOGY_DIR,
            ROOT / "ProcessedData/feature_schema.json",
            CACHE_DIR,
            24,
        )
        train, validation, test = (
            bundle.by_split(split) for split in ("train", "validation", "test")
        )
        scaler = FeatureScaler(bundle.feature_names).fit(train)
        scaler.transform(bundle.examples)

        candidates = []
        for config_index, config in enumerate(CONFIGS):
            started = time.time()
            model, score, epoch, history = fit_config(
                config,
                train,
                validation,
                len(bundle.feature_names),
                SEED + horizon_index * 100 + config_index,
            )
            leaderboard.append(
                {
                    "task": task,
                    **config,
                    "validation_auprc": float(score),
                    "best_epoch": int(epoch),
                    "parameter_n": sum(parameter.numel() for parameter in model.parameters()),
                    "fit_seconds": time.time() - started,
                }
            )
            pd.DataFrame(history).to_csv(
                OUT / f"{task}_{config['name']}_history.csv", index=False
            )
            candidates.append((score, config, model, epoch))

        score, config, model, epoch = max(candidates, key=lambda item: item[0])
        validation_probability = predict(model, validation)
        test_probability = predict(model, test)
        y_test = labels(test)

        xgb = pd.read_csv(
            BASE_RESULTS
            / "predictions"
            / f"{task}_structured_xgboost_test.csv",
            dtype={"visit_no": str},
        ).set_index("visit_no")
        visit_order = [str(example.visit_no) for example in test]
        xgb_probability = xgb.loc[visit_order, "probability"].to_numpy()
        comparison = paired_delta_ci(
            y_test, test_probability, xgb_probability, SEED + 1000 + horizon
        )
        result = {
            "task": task,
            "selected_config": config["name"],
            "validation_auprc": float(score),
            "best_epoch": int(epoch),
            **metrics(y_test, test_probability),
            **comparison,
        }
        selected_rows.append(result)
        pd.DataFrame(
            {
                "patient_id": [example.patient_id for example in test],
                "visit_no": visit_order,
                "label": y_test,
                "probability": test_probability,
            }
        ).to_csv(OUT / f"{task}_selected_structured_transformer_test.csv", index=False)
        torch.save(
            {
                "task": task,
                "variant": "structured",
                "config": config,
                "state_dict": model.state_dict(),
                "feature_names": bundle.feature_names,
                "scaler": scaler.state_dict(),
                "best_epoch": int(epoch),
                "validation_auprc": float(score),
            },
            OUT / f"{task}_selected_structured_transformer.pt",
        )

        # Release models not selected for the locked test evaluation.
        del candidates, model

    pd.DataFrame(leaderboard).to_csv(OUT / "validation_leaderboard.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUT / "selected_test_results.csv", index=False)
    (OUT / "selection_protocol.json").write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "test_used_for_selection": False,
                "selection_metric": "AUPRC",
                "config_names": [config["name"] for config in CONFIGS],
                "seed_base": SEED,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(selected_rows).to_string(index=False))


if __name__ == "__main__":
    main()
