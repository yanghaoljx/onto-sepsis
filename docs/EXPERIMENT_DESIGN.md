# Irregular-time sepsis prediction experiment design

## 1. Data split and reporting hierarchy

The existing patient-level split is retained:

- `train`: model fitting;
- `validation`: model selection and the primary reported performance;
- `test`: locked independent confirmation after the model and settings have been selected.

The test set should not be used to select epochs, thresholds, features, or model variants. The main manuscript table should therefore report validation AUROC/AUPRC and operating-point metrics, while the test table should be labelled independent confirmation.

This reporting structure is consistent with the emphasis on transparent development and evaluation in [TRIPOD+AI](https://www.bmj.com/content/385/bmj.q902). For deployment-oriented context, the prospective multi-site evaluation of TREWS in [Nature Medicine](https://www.nature.com/articles/s41591-022-01894-0) is a useful example of separating model development from real-world outcome assessment.

## 2. Prediction endpoint and irregular sequence construction

The input is not resampled to a regular grid. For each admission and each horizon:

1. Positive endpoint: select the observed event-time row closest to 6, 12, or 24 hours before sepsis onset, subject to the row being before the onset.
2. Negative endpoint: select one pseudo-anchor per negative admission. Its admission-relative time is deterministically matched to the **training-positive** endpoint-time distribution and the same reference distribution is used for validation and test. This avoids treating every negative round as an independent patient, prevents the negative class from being dominated by long admissions, and keeps test outcomes out of index-time construction.
3. Prefix: retain only rows from the same admission with `round_time <= endpoint_time`, and keep the most recent `max_seq_len` events.
4. Time representation: retain the original event times and add `delta_hours` between consecutive events and `hours_from_admission`. These are continuous covariates, not regularly spaced bins.

The code therefore supports one patient with 3 events and another with 24 events without pretending that either has measurements at unobserved times. The endpoint label is attached only to the final row of the prefix; outcome-related columns such as `hours_to_sepsis`, `anchor_distance_hours`, and `label` are never passed as model features.

The negative pseudo-anchor is a modelling convention that should be stated explicitly in the Methods. If the study team later defines a clinical index time for negative admissions, replace `select_endpoints()` in `temporal_data.py` with that prespecified rule and rerun every arm.

## 3. Seven-model comparison

The main matrix is 3 horizons × 2 feature variants × 7 models:

| Category | Model | Role |
|---|---|---|
| Tabular baseline | Logistic regression | Transparent, calibrated reference |
| Tabular baseline | Random forest | Nonlinear ensemble reference |
| Tabular baseline | XGBoost | Strong structured-data benchmark |
| Sequential | BiLSTM | Established recurrent sequence baseline |
| Sequential | BiGRU | Lower-parameter recurrent baseline |
| Irregular-time sequential | GRU-D | Explicit decay for missingness and elapsed time |
| Irregular-time sequential | Time-aware Transformer | Attention with continuous event-time encoding |

GRU-D is motivated by [Che et al., Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://arxiv.org/abs/1606.01865), and the broader treatment of irregular sampling as a missing-data problem follows [Li and Marlin](https://proceedings.mlr.press/v119/li20k.html). The Transformer arm is not given a fake regular grid: its time embedding uses the observed interval and admission-relative time.

## 4. Feature ablation

- `structured`: numeric laboratory/vital features and their measurement masks, plus age if present and the two continuous time features.
- `state`: structured features plus ontology-derived clinical state features only.
- `relation`: structured features plus ontology relation-count features only.
- `ontology`: the complete 199-dimensional feature vector, including structured measurements, clinical state features, and ontology relation counts, plus the same time features.

The main run compares `structured` and `ontology`. The extended ablation is available with `--variant extended` and separates state and relation contributions. All arms use identical endpoints, splits, sequence truncation, optimizer settings, and evaluation code. The primary ablation is `ontology - structured` for AUPRC, because the positive class is not balanced.

The main tabular baseline uses the latest observed event at the prediction endpoint (`--tabular-pooling last`), which is straightforward to deploy in a real-time setting. The previous `last + mean + min + max` representation remains available with `--tabular-pooling summary` as a sensitivity analysis; it is richer and may give tree models an additional advantage. A deep model with attention or pooled hidden states can then be added as a further sensitivity analysis.

## 5. Metrics and figures

Primary metrics:

- AUROC and AUPRC;
- sensitivity at 90% and 95% specificity;
- specificity, PPV, NPV, F1 at a prespecified threshold;
- Brier score, calibration slope, and calibration intercept.

The generated figures are:

- `Figure_2_performance_heatmaps`: validation primary results and test confirmation across horizons/models;
- `Figure_3_validation_curves`: ROC and precision–recall curves for the best validation model in each feature arm;
- `Figure_4_validation_calibration`: calibration curves;
- `Figure_5_ontology_ablation`: matched change in validation AUPRC after ontology enhancement.
- `Figure_6_sequence_strata`: performance by sequence length and endpoint-time strata.

The accompanying analyses are:

- `analyze_ontology_effect.py`: paired admission-level bootstrap for ontology versus structured predictions on validation and test;
- `analyze_sequence_strata.py`: performance by number of observed events and admission-relative endpoint time;
- `audit_ontology_leakage.py`: deterministic audit for outcome words in accepted evidence and observations after the prediction cutoff;
- `--variant extended`: state-only versus relation-only ontology ablations.

These analyses help distinguish three possible explanations for an ontology gain: added clinically meaningful state information, added relation-count information, or merely a change in the amount of effective information available at the endpoint. Feature-level explanation should be performed on the locked XGBoost model using grouped feature importance or permutation importance; the primary evidence should remain paired performance and confidence intervals rather than post-hoc feature anecdotes.

For the manuscript, add 95% confidence intervals using admission-level bootstrap resampling of the locked validation and test prediction files. Do not bootstrap individual event rows, because one admission contributes a variable number of irregular observations.
