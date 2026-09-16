# SEPON: Sepsis Early Prediction with Ontology-Aware Temporal Models

This repository contains the reproducible code and ontology specification for the SEPON sepsis early-prediction pipeline. It intentionally contains **no patient-level data, database dump, trained model, or manuscript source file**.

## Included

- Temporal feature construction and model training utilities (`src/`)
- Sepsis prediction ontology (`ontology/`)
- Contract and preprocessing tests (`tests/`)
- MIMIC-IV extraction templates and external-validation notes (`docs/`)
- One non-sensitive example figure (`examples/`)

## Data policy

Patient-level data, `Dataset.zip`, `ProcessedData/`, row-level MIMIC extracts, generated result tables, model checkpoints, and manuscript files are excluded by design. Obtain MIMIC-IV through PhysioNet and follow its data-use agreement. Keep all raw and derived row-level files outside this repository.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_temporal.txt
```

Review and adapt local input paths before running any script. See [`docs/MIMIC_EXTERNAL_VALIDATION.md`](docs/MIMIC_EXTERNAL_VALIDATION.md) for cohort definitions and SQL templates.

## Tests

```bash
python -m pytest -q tests
```

## Example workflow

```bash
python src/build_feature_schema.py
python src/build_timepoint_dataset.py
python src/run_temporal_experiments.py
```

These are research utilities, not a packaged CLI. Results depend on the local data snapshot, preprocessing choices, software versions, and approved MIMIC-IV access. Ontology-derived features require a leakage audit before use in a new cohort.

## Citation and license

Cite the associated SEPON manuscript when publicly available and include the repository commit hash used for analysis. Add a software license matching your institutional and data-use policy before publishing. A software license does not grant permission to redistribute MIMIC-IV or other restricted clinical data.
