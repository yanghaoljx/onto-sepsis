"""Run the irregular-time sepsis prediction experiment matrix.

Primary reporting split: validation.
Independent confirmation split: test.

Examples
--------
Local smoke test (CPU, a few visits per split)::

    python run_temporal_experiments.py \
      --task pre_6h --variant structured --model gru_d \
      --max-visits-per-split 20 --epochs 1 --batch-size 8 \
      --device cpu --output-dir ProcessedData/temporal_results/smoke

Server full matrix on an H100::

    python run_temporal_experiments.py \
      --task all --variant all --model all \
      --device cuda --batch-size 256 --epochs 20 --num-workers 4 \
      --output-dir ProcessedData/temporal_results/main
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from temporal_data import (
    TASKS,
    FeatureScaler,
    SequenceBundle,
    SequenceExample,
    endpoint_matrix,
    endpoint_feature_names,
    labels,
    build_bundle,
    metadata_rows,
)
from temporal_models import (
    MODEL_NAMES,
    SequenceDataset,
    collate_sequences,
    make_sequence_model,
)


TABULAR_MODELS = ("logreg", "random_forest", "xgboost")
ALL_MODELS = TABULAR_MODELS + MODEL_NAMES
MAIN_VARIANTS = ("structured", "ontology")
EXTENDED_VARIANTS = ("structured", "state", "relation", "ontology")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return device


def _safe_metric(function: Any, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(function(y_true, y_score))
    except ValueError:
        return float("nan")


def _binary_at_threshold(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (probability >= threshold).astype(np.int64)
    tp = float(np.sum((predicted == 1) & (y_true == 1)))
    tn = float(np.sum((predicted == 0) & (y_true == 0)))
    fp = float(np.sum((predicted == 1) & (y_true == 0)))
    fn = float(np.sum((predicted == 0) & (y_true == 1)))
    return {
        "sensitivity": tp / max(tp + fn, 1.0),
        "specificity": tn / max(tn + fp, 1.0),
        "ppv": tp / max(tp + fp, 1.0),
        "npv": tn / max(tn + fn, 1.0),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
    }


def _threshold_at_specificity(
    y_true: np.ndarray,
    probability: np.ndarray,
    target_specificity: float,
) -> dict[str, float]:
    threshold = _select_threshold_at_specificity(y_true, probability, target_specificity)
    values = _binary_at_threshold(y_true, probability, threshold)
    return {
        f"sensitivity_at_specificity_{int(target_specificity * 100)}": values["sensitivity"],
        f"specificity_at_specificity_{int(target_specificity * 100)}": values["specificity"],
        f"threshold_at_specificity_{int(target_specificity * 100)}": threshold,
    }


def _select_threshold_at_specificity(
    y_true: np.ndarray,
    probability: np.ndarray,
    target_specificity: float,
) -> float:
    thresholds = np.unique(np.concatenate(([1.0], probability, [0.0])))
    candidates: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        values = _binary_at_threshold(y_true, probability, float(threshold))
        if values["specificity"] >= target_specificity:
            candidates.append((values["sensitivity"], values["specificity"], float(threshold)))
    if not candidates:
        threshold = 1.0
    else:
        _, _, threshold = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
    return float(threshold)


def _calibration(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        model.fit(logit, y_true)
        return float(model.coef_[0, 0]), float(model.intercept_[0])
    except ValueError:
        return float("nan"), float("nan")


def calculate_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    operating_thresholds: dict[int, float] | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    calibration_slope, calibration_intercept = _calibration(y_true, probability)
    metrics = {
        "n": float(len(y_true)),
        "positive_n": float(y_true.sum()),
        "negative_n": float((1 - y_true).sum()),
        "auroc": _safe_metric(roc_auc_score, y_true, probability),
        "auprc": _safe_metric(average_precision_score, y_true, probability),
        "brier": float(brier_score_loss(y_true, probability)),
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
    }
    metrics.update(_binary_at_threshold(y_true, probability, 0.5))
    for target in (90, 95):
        if operating_thresholds is None:
            metrics.update(_threshold_at_specificity(y_true, probability, target / 100.0))
        else:
            threshold = operating_thresholds[target]
            values = _binary_at_threshold(y_true, probability, threshold)
            metrics.update(
                {
                    f"sensitivity_at_specificity_{target}": values["sensitivity"],
                    f"specificity_at_specificity_{target}": values["specificity"],
                    f"threshold_at_specificity_{target}": threshold,
                }
            )
    return metrics


def _class_weight_ratio(y: np.ndarray) -> float:
    positive = max(int(y.sum()), 1)
    negative = max(int((1 - y).sum()), 1)
    return negative / positive


def _make_tabular_model(name: str, y_train: np.ndarray, seed: int) -> Any:
    if name == "logreg":
        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=500,
            solver="lbfgs",
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "xgboost is required for the xgboost arm; install requirements_temporal.txt"
            ) from exc
        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=_class_weight_ratio(y_train),
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown tabular model: {name}")


def _predict_model(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    logits = model(x)
    return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)


def train_tabular(
    model_name: str,
    bundle: SequenceBundle,
    train_examples: Sequence[SequenceExample],
    validation_examples: Sequence[SequenceExample],
    test_examples: Sequence[SequenceExample],
    seed: int,
    pooling: str,
    save_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    x_train = endpoint_matrix(train_examples, pooling=pooling)
    x_validation = endpoint_matrix(validation_examples, pooling=pooling)
    x_test = endpoint_matrix(test_examples, pooling=pooling)
    y_train = labels(train_examples)
    model = _make_tabular_model(model_name, y_train, seed)
    started = time.time()
    model.fit(x_train, y_train)
    if save_path is not None:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError(
                "joblib is required when --save-tabular-models is used"
            ) from exc
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
    elapsed = time.time() - started
    probabilities = {
        "train": _predict_model(model, x_train),
        "validation": _predict_model(model, x_validation),
        "test": _predict_model(model, x_test),
    }
    return probabilities, {
        "fit_seconds": elapsed,
        "input_dim": int(x_train.shape[1]),
        "tabular_pooling": pooling,
        "tabular_feature_names": endpoint_feature_names(bundle.feature_names, pooling),
    }


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def predict_sequence_model(
    model: nn.Module,
    examples: Sequence[SequenceExample],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    loader = DataLoader(
        SequenceDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequences,
    )
    model.eval()
    outputs: list[np.ndarray] = []
    for batch in loader:
        tensors = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in batch.items()
        }
        with _autocast_context(device):
            logits = model(
                tensors["x"],
                tensors["observed_mask"],
                tensors["delta_hours"],
                tensors["lengths"],
            )
        outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def train_sequence(
    model_name: str,
    train_examples: Sequence[SequenceExample],
    validation_examples: Sequence[SequenceExample],
    test_examples: Sequence[SequenceExample],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    num_workers: int,
    seed: int,
    save_path: Path | None = None,
    artifact_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    # The seed must control initialization/dropout as well as the DataLoader.
    # Otherwise resuming or changing model order changes the fitted model.
    seed_everything(seed)
    model = make_sequence_model(model_name, train_examples[0].x.shape[1], hidden_dim=hidden_dim,
                                recurrent_pooling="final_hidden")
    model.to(device)
    y_train = labels(train_examples)
    pos_weight = torch.tensor([_class_weight_ratio(y_train)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loader = DataLoader(
        SequenceDataset(train_examples),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequences,
        generator=torch.Generator().manual_seed(seed),
    )

    best_state: dict[str, Tensor] | None = None
    best_auprc = -float("inf")
    best_epoch = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            tensors = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device):
                logits = model(
                    tensors["x"],
                    tensors["observed_mask"],
                    tensors["delta_hours"],
                    tensors["lengths"],
                )
                loss = criterion(logits.float(), tensors["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_probability = predict_sequence_model(
            model, validation_examples, device, batch_size, num_workers
        )
        validation_auprc = calculate_metrics(labels(validation_examples), validation_probability)["auprc"]
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"  {model_name} epoch {epoch:02d}/{epochs}: "
            f"train_loss={mean_loss:.4f} validation_AUPRC={validation_auprc:.4f}",
            flush=True,
        )
        if np.isfinite(validation_auprc) and validation_auprc > best_auprc:
            best_auprc = validation_auprc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"No valid validation checkpoint was produced for {model_name}")
    model.load_state_dict(best_state)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": model_name,
                "input_dim": int(train_examples[0].x.shape[1]),
                "hidden_dim": int(hidden_dim),
                "best_epoch": int(best_epoch),
                "best_validation_auprc": float(best_auprc),
                "state_dict": best_state,
                **(artifact_metadata or {}),
                "recurrent_pooling": "final_hidden" if model_name in {"lstm", "gru"} else None,
                "seed": int(seed),
            },
            save_path,
        )
    probabilities = {
        "train": predict_sequence_model(model, train_examples, device, batch_size, num_workers),
        "validation": predict_sequence_model(model, validation_examples, device, batch_size, num_workers),
        "test": predict_sequence_model(model, test_examples, device, batch_size, num_workers),
    }
    fit_info = {
        "fit_seconds": time.time() - started,
        "best_epoch": best_epoch,
        "best_validation_auprc": best_auprc,
        "input_dim": int(train_examples[0].x.shape[1]),
        "device": str(device),
        "seed": int(seed),
        "recurrent_pooling": "final_hidden" if model_name in {"lstm", "gru"} else None,
    }
    del model, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities, fit_info


def _write_predictions(path: Path, examples: Sequence[SequenceExample], probability: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = metadata_rows(examples)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "patient_id",
                "visit_no",
                "split",
                "label",
                "endpoint_time",
                "sequence_length",
                "hours_from_admission",
                "probability",
            ],
        )
        writer.writeheader()
        for row, value in zip(rows, probability):
            row["probability"] = float(value)
            writer.writerow(row)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "all"], default="all")
    parser.add_argument(
        "--variant",
        choices=[*EXTENDED_VARIANTS, "all", "extended"],
        default="all",
        help="all runs structured+ontology; extended adds state-only and relation-only arms",
    )
    parser.add_argument(
        "--model",
        choices=[*ALL_MODELS, "all", "comparators"],
        default="all",
        help="comparators runs all models except the separately tuned Transformer",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("ProcessedData/model_dataset/pre_sepsis_anchors"),
    )
    parser.add_argument(
        "--ontology-dir",
        type=Path,
        default=Path("ProcessedData/model_dataset/ontology_enhanced"),
    )
    parser.add_argument("--schema-path", type=Path, default=Path("ProcessedData/feature_schema.json"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("ProcessedData/model_dataset/temporal_cache"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("ProcessedData/temporal_results/main"))
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--tabular-pooling",
        choices=["summary", "last"],
        default="last",
        help="last=latest observed event at the endpoint; summary=last+mean+min+max sensitivity audit",
    )
    parser.add_argument(
        "--save-tabular-models",
        action="store_true",
        help="save fitted sklearn/XGBoost models under output-dir/model_artifacts",
    )
    parser.add_argument(
        "--save-sequence-models",
        action="store_true",
        help="save the best validation checkpoint and preprocessing metadata for deep models",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-visits-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _requested(value: str, all_values: Sequence[str]) -> list[str]:
    return list(all_values) if value == "all" else [value]


def _requested_variants(value: str) -> list[str]:
    if value == "all":
        return list(MAIN_VARIANTS)
    if value == "extended":
        return list(EXTENDED_VARIANTS)
    return [value]


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    tasks = _requested(args.task, TASKS)
    variants = _requested_variants(args.variant)
    models = (
        [model for model in ALL_MODELS if model != "transformer"]
        if args.model == "comparators"
        else _requested(args.model, ALL_MODELS)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    existing_config = args.output_dir / "run_config.json"
    if metrics_path.exists() and not args.overwrite:
        previous_config = json.loads(existing_config.read_text()) if existing_config.exists() else {}
        if previous_config.get("preprocessing_version") != 3:
            raise ValueError("Existing results predate numeric mapping/scaling/pooling repairs. "
                             "Choose a new --output-dir; do not resume a mixed-version experiment.")
    if args.overwrite and metrics_path.exists():
        metrics_path.unlink()
    completed_keys: set[tuple[str, str, str, str]] = set()
    records_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if metrics_path.exists() and not args.overwrite:
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (
                str(record.get("task")),
                str(record.get("variant")),
                str(record.get("model")),
                str(record.get("split")),
            )
            completed_keys.add(key)
            records_by_key[key] = record
    run_config = vars(args).copy()
    run_config.update(
        {
            "device_resolved": str(device),
            "tasks": tasks,
            "variants": variants,
            "models": models,
            "primary_split": "validation",
            "independent_split": "test",
            "preprocessing_version": 3,
            "recurrent_pooling": "final_hidden",
            "endpoint_definition": "one endpoint per visit; prefix only; negative pseudo-anchor matched to the train-positive admission-time distribution",
        }
    )
    _write_json(args.output_dir / "run_config.json", run_config)

    for task in tasks:
        for variant in variants:
            print(f"\n=== {task} / {variant} ===", flush=True)
            bundle = build_bundle(
                task=task,
                variant=variant,
                data_dir=args.data_dir,
                ontology_dir=args.ontology_dir,
                schema_path=args.schema_path,
                cache_dir=args.cache_dir,
                max_seq_len=args.max_seq_len,
                max_rows=args.max_rows,
                max_visits_per_split=args.max_visits_per_split,
            )
            train_examples = bundle.by_split("train")
            validation_examples = bundle.by_split("validation")
            test_examples = bundle.by_split("test")
            if not train_examples or not validation_examples or not test_examples:
                raise RuntimeError(
                    f"Empty split in {task}/{variant}: "
                    f"train={len(train_examples)}, validation={len(validation_examples)}, test={len(test_examples)}"
                )
            split_counts = {
                split: {
                    "n": len(examples),
                    "positive": int(labels(examples).sum()),
                    "negative": int(len(examples) - labels(examples).sum()),
                }
                for split, examples in (
                    ("train", train_examples),
                    ("validation", validation_examples),
                    ("test", test_examples),
                )
            }
            print(f"  endpoint counts: {json.dumps(split_counts, ensure_ascii=False)}", flush=True)
            invalid_splits = [
                split
                for split, counts in split_counts.items()
                if counts["positive"] == 0 or counts["negative"] == 0
            ]
            if invalid_splits:
                raise RuntimeError(
                    f"{task}/{variant} has only one class in {invalid_splits}: {split_counts}. "
                    "For a smoke test, remove --max-rows or increase it until every split "
                    "contains both labels; never run the final experiment with a single-class split."
                )
            scaler = FeatureScaler(bundle.feature_names).fit(train_examples)
            scaler.transform(bundle.examples)
            _write_json(
                args.output_dir / f"{task}_{variant}_dataset_summary.json",
                {
                    "task": task,
                    "variant": variant,
                    "feature_dim": len(bundle.feature_names),
                    "feature_names": bundle.feature_names,
                    "n_train": len(train_examples),
                    "n_validation": len(validation_examples),
                    "n_test": len(test_examples),
                    "positive_train": int(labels(train_examples).sum()),
                    "positive_validation": int(labels(validation_examples).sum()),
                    "positive_test": int(labels(test_examples).sum()),
                    "sequence_length": {
                        split: {
                            "min": int(min(example.length for example in bundle.by_split(split))),
                            "median": float(np.median([example.length for example in bundle.by_split(split)])),
                            "max": int(max(example.length for example in bundle.by_split(split))),
                        }
                        for split in ("train", "validation", "test")
                    },
                    "scaler": scaler.state_dict(),
                },
            )

            for model_name in models:
                prediction_paths = [
                    args.output_dir / "predictions" / f"{task}_{variant}_{model_name}_{split}.csv"
                    for split in ("train", "validation", "test")
                ]
                arm_complete = all(
                    (task, variant, model_name, split) in completed_keys and path.exists()
                    for split, path in zip(("train", "validation", "test"), prediction_paths)
                )
                if arm_complete and not args.overwrite:
                    print(f"  skip completed {model_name} (resume)", flush=True)
                    continue
                print(f"  fitting {model_name} ...", flush=True)
                try:
                    if model_name in TABULAR_MODELS:
                        probabilities, fit_info = train_tabular(
                            model_name,
                            bundle,
                            train_examples,
                            validation_examples,
                            test_examples,
                            args.seed,
                            args.tabular_pooling,
                            (
                                args.output_dir
                                / "model_artifacts"
                                / f"{task}_{variant}_{model_name}.joblib"
                            )
                            if args.save_tabular_models
                            else None,
                        )
                    else:
                        probabilities, fit_info = train_sequence(
                            model_name,
                            train_examples,
                            validation_examples,
                            test_examples,
                            device,
                            args.epochs,
                            args.batch_size,
                            args.learning_rate,
                            args.weight_decay,
                            args.hidden_dim,
                            args.num_workers,
                            args.seed,
                            (
                                args.output_dir
                                / "model_artifacts"
                                / f"{task}_{variant}_{model_name}.pt"
                            )
                            if args.save_sequence_models
                            else None,
                            {
                                "task": task,
                                "variant": variant,
                                "feature_names": bundle.feature_names,
                                "scaler": scaler.state_dict(),
                                "max_seq_len": args.max_seq_len,
                                "preprocessing_version": 3,
                            },
                        )
                except RuntimeError as exc:
                    if model_name == "xgboost" and "required" in str(exc):
                        print(f"  SKIP {model_name}: {exc}", flush=True)
                        continue
                    raise

                validation_probability = probabilities["validation"]
                operating_thresholds = {
                    target: _select_threshold_at_specificity(
                        labels(validation_examples), validation_probability, target / 100.0
                    )
                    for target in (90, 95)
                }
                for split, examples in (
                    ("train", train_examples),
                    ("validation", validation_examples),
                    ("test", test_examples),
                ):
                    probability = probabilities[split]
                    metrics = calculate_metrics(
                        labels(examples), probability, operating_thresholds=operating_thresholds
                    )
                    record = {
                        "task": task,
                        "variant": variant,
                        "model": model_name,
                        "split": split,
                        "primary_split": split == "validation",
                        "independent_split": split == "test",
                        "operating_threshold_source": "validation",
                        **metrics,
                        **fit_info,
                    }
                    key = (task, variant, model_name, split)
                    records_by_key[key] = record
                    completed_keys.add(key)
                    with metrics_path.open("w", encoding="utf-8") as handle:
                        for saved_record in records_by_key.values():
                            handle.write(json.dumps(saved_record, ensure_ascii=False) + "\n")
                    _write_predictions(
                        args.output_dir / "predictions" / f"{task}_{variant}_{model_name}_{split}.csv",
                        examples,
                        probability,
                    )
                print(
                    f"  done {model_name}: validation AUPRC="
                    f"{calculate_metrics(labels(validation_examples), probabilities['validation'], operating_thresholds)['auprc']:.4f}, "
                    f"test AUPRC="
                    f"{calculate_metrics(labels(test_examples), probabilities['test'], operating_thresholds)['auprc']:.4f}",
                    flush=True,
                )

    print(f"\nFinished. Metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
