# MIMIC-IV external-validation extract

This directory contains a read-only, reproducible extract aligned to the
project's **structured numeric** feature arm. It does not modify the MIMIC-IV
database.

## Cohort and index times

- Adults (`admission_age >= 18`), first ICU stay of the first hospital stay.
- Positive: MIMIC-IV `mimiciv_derived.sepsis3`; prediction time is 6, 12, or
  24 hours before `suspected_infection_time`.
- Negative: no row in `sepsis3`; the pseudo-index time is the median
  ICU-relative positive index time for that horizon.
- Both classes must still be in the ICU at the index time.
- The tuned temporal-transfer analysis additionally restricts the external
  cohort to sequence length ≥8 and numeric-feature completeness ≥70%.
  Source events are aggregated into 4-hour admission-relative buckets with
  cumulative forward filling, yielding a variable-length prefix that is
  compatible with the local Transformer while retaining both outcome classes.
- No class balancing or fixed-size sampling is applied in the reported primary
  external validation.
- Each source measurement is retained up to the index time; the endpoint-only
  extract remains available for the tabular structured baseline.

## Run

From the repository root:

```bash
mkdir -p ProcessedData/mimic_external_validation
PGPASSWORD='...' psql -h localhost -U postgres -d mimiciv \
  -f mimic_external_validation/extract_mimic_endpoint.sql
```

The generated CSVs contain no direct identifiers beyond MIMIC's deidentified
integer IDs. Keep them under the same data-use controls as the source MIMIC
database and do not commit row-level extracts to a public repository.

## Important limitations

- `value__lab__procalcitonin` is unavailable in the installed MIMIC derived
  tables and is emitted as null with mask 0.
- CRP is available but sparse. Missingness is retained; rows are not selected
  for completeness, avoiding complete-case selection bias.
- MIMIC stores BUN in mg/dL; `value__lab__urea` is converted to mmol/L using
  `BUN * 0.357` to match the local feature's likely convention. Confirm the
  local raw unit before locked evaluation.
- The MIMIC `sepsis3` onset proxy is not identical to the local outcome
  adjudication. This must be reported as a domain/label-definition difference.
- Only the 37 structured values and their masks are harmonized here. Ontology
  state/relation features require a separate, leakage-audited diagnosis,
  treatment, microbiology, and note mapping.
- The negative pseudo-index is fixed at the horizon-specific median positive
  ICU-relative index time. This avoids post-index data but compresses the
  negative time distribution and remains a sensitivity-analysis limitation.
