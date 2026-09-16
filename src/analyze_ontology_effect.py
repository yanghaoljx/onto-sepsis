"""Paired validation/test analysis of ontology-enhanced predictions.

The structured and ontology arms use the same admission endpoints.  This
script therefore uses the same bootstrap admission indices for both arms and
reports paired changes rather than comparing two unrelated confidence
intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


TASKS = ("pre_6h", "pre_12h", "pre_24h")
MODELS = ("logreg", "random_forest", "xgboost", "lstm", "gru", "gru_d", "transformer")
SPLITS = ("validation", "test")


def _metric(y: np.ndarray, p: np.ndarray, name: str) -> float:
    if name in {"auprc", "auroc"} and len(np.unique(y)) < 2:
        return float("nan")
    if name == "auprc":
        return float(average_precision_score(y, p))
    if name == "auroc":
        return float(roc_auc_score(y, p))
    if name == "brier":
        return float(brier_score_loss(y, p))
    raise ValueError(name)


def _prediction_path(results_dir: Path, task: str, variant: str, model: str, split: str) -> Path:
    return results_dir / "predictions" / f"{task}_{variant}_{model}_{split}.csv"


def paired_bootstrap(
    y: np.ndarray,
    structured: np.ndarray,
    ontology: np.ndarray,
    metric: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    observed = _metric(y, ontology, metric) - _metric(y, structured, metric)
    values = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y), size=len(y))
        sampled_y = y[index]
        values.append(
            _metric(sampled_y, ontology[index], metric)
            - _metric(sampled_y, structured[index], metric)
        )
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return observed, float("nan"), float("nan")
    return observed, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("ProcessedData/temporal_results/main"))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    records = []
    for task in TASKS:
        for model in MODELS:
            for split in SPLITS:
                structured_path = _prediction_path(args.results_dir, task, "structured", model, split)
                ontology_path = _prediction_path(args.results_dir, task, "ontology", model, split)
                if not structured_path.exists() or not ontology_path.exists():
                    continue
                structured = pd.read_csv(structured_path)
                ontology = pd.read_csv(ontology_path)
                if not structured["visit_no"].astype(str).equals(ontology["visit_no"].astype(str)):
                    raise ValueError(f"Endpoint ordering differs for {task}/{model}/{split}")
                y = structured["label"].to_numpy(dtype=np.int64)
                p_structured = structured["probability"].to_numpy(dtype=float)
                p_ontology = ontology["probability"].to_numpy(dtype=float)
                record = {"task": task, "model": model, "split": split}
                for metric in ("auprc", "auroc", "brier"):
                    delta, lower, upper = paired_bootstrap(
                        y, p_structured, p_ontology, metric, rng, args.n_bootstrap
                    )
                    record.update(
                        {
                            f"structured_{metric}": _metric(y, p_structured, metric),
                            f"ontology_{metric}": _metric(y, p_ontology, metric),
                            f"delta_{metric}": delta,
                            f"delta_{metric}_lower": lower,
                            f"delta_{metric}_upper": upper,
                        }
                    )
                records.append(record)

    output_path = args.results_dir / "ontology_paired_effects.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Paired ontology effects written to {output_path}")


if __name__ == "__main__":
    main()

