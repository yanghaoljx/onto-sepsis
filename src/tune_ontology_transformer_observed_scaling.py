#!/usr/bin/env python3
"""Run ontology Transformer tuning with observed-value-only scaling."""

from pathlib import Path

import tune_time_aware_transformer as tuning


if __name__ == "__main__":
    tuning.OUT = (
        Path(__file__).resolve().parent
        / "ProcessedData/temporal_results/transformer_validation_tuning_carryforward_median"
    )
    tuning.main()
