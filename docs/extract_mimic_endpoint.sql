\set ON_ERROR_STOP on

-- Source tables are queried only. The only database objects created are
-- session-local temporary tables, automatically discarded on disconnect.
SET statement_timeout = 0;

CREATE TEMP TABLE seppred_candidates AS
WITH first_stays AS (
    SELECT d.*
    FROM mimiciv_derived.icustay_detail d
    WHERE d.admission_age >= 18
      AND d.first_hosp_stay
      AND d.first_icu_stay
), labeled AS (
    SELECT f.*,
           s.suspected_infection_time AS sepsis_time,
           (s.stay_id IS NOT NULL) AS label
    FROM first_stays f
    LEFT JOIN mimiciv_derived.sepsis3 s USING (subject_id, stay_id)
), horizons AS (
    SELECT unnest(ARRAY[6, 12, 24])::integer AS horizon_h
), positive AS (
    SELECT l.*, h.horizon_h,
           l.sepsis_time - make_interval(hours => h.horizon_h) AS prediction_time
    FROM labeled l CROSS JOIN horizons h
    WHERE l.label
      AND l.sepsis_time - make_interval(hours => h.horizon_h) >= l.icu_intime
      AND l.sepsis_time - make_interval(hours => h.horizon_h) <= l.icu_outtime
), median_index AS (
    SELECT horizon_h,
           percentile_cont(0.5) WITHIN GROUP
             (ORDER BY extract(epoch FROM (prediction_time - icu_intime))/3600.0)
             AS median_icu_hour
    FROM positive
    GROUP BY horizon_h
), negative AS (
    SELECT l.*, h.horizon_h,
           l.icu_intime + make_interval(secs => m.median_icu_hour * 3600.0)
             AS prediction_time
    FROM labeled l
    CROSS JOIN horizons h
    JOIN median_index m USING (horizon_h)
    WHERE NOT l.label
      AND l.icu_intime + make_interval(secs => m.median_icu_hour * 3600.0)
          <= l.icu_outtime
)
-- Use every eligible ICU stay.  The former deterministic cap of 100 stays per
-- class was useful only for pipeline debugging; it created an artificial 1:1
-- case-control cohort and is not appropriate for the reported validation.
SELECT * FROM positive
UNION ALL
SELECT * FROM negative;

CREATE INDEX ON seppred_candidates(subject_id, hadm_id, stay_id, prediction_time);

-- ``mimiciv_derived.vitalsign`` may not have an index on stay/time.  Restrict
-- it once to the relevant stays and index the session-local copy so that the
-- all-stay extraction does not rescan the full table for every endpoint.
CREATE TEMP TABLE seppred_vitalsign AS
SELECT v.*
FROM mimiciv_derived.vitalsign v
JOIN (SELECT DISTINCT stay_id FROM seppred_candidates) c USING (stay_id);
CREATE INDEX ON seppred_vitalsign(stay_id, charttime);

CREATE TEMP TABLE seppred_endpoint AS
WITH endpoint AS (
SELECT
    c.subject_id, c.hadm_id, c.stay_id, c.horizon_h,
    c.label::integer AS label, c.icu_intime, c.prediction_time, c.sepsis_time,
    c.admission_age::double precision AS value__demographic__age,
    1 AS mask__value__demographic__age,
    v.heart_rate AS value__vital__heart_rate,
    (v.heart_rate IS NOT NULL)::integer AS mask__value__vital__heart_rate,
    v.sbp AS value__vital__systolic_bp,
    (v.sbp IS NOT NULL)::integer AS mask__value__vital__systolic_bp,
    v.dbp AS value__vital__diastolic_bp,
    (v.dbp IS NOT NULL)::integer AS mask__value__vital__diastolic_bp,
    v.mbp AS value__vital__mean_bp,
    (v.mbp IS NOT NULL)::integer AS mask__value__vital__mean_bp,
    v.resp_rate AS value__vital__respiratory_rate,
    (v.resp_rate IS NOT NULL)::integer AS mask__value__vital__respiratory_rate,
    v.temperature::double precision AS value__vital__temperature,
    (v.temperature IS NOT NULL)::integer AS mask__value__vital__temperature,
    v.spo2 AS value__vital__oxygen_saturation,
    (v.spo2 IS NOT NULL)::integer AS mask__value__vital__oxygen_saturation,
    v.glucose AS value__vital__glucose,
    (v.glucose IS NOT NULL)::integer AS mask__value__vital__glucose,
    ch.albumin AS value__lab__albumin,
    ch.bicarbonate AS value__lab__hco3,
    ch.bun * 0.357 AS value__lab__urea,
    ch.calcium AS value__lab__calcium,
    ch.chloride AS value__lab__chloride,
    ch.creatinine AS value__lab__creatinine,
    ch.glucose AS value__lab__glucose,
    ch.sodium AS value__lab__sodium,
    ch.potassium AS value__lab__potassium,
    cb.hemoglobin AS value__lab__hemoglobin,
    cb.platelet AS value__lab__platelet,
    cb.wbc AS value__lab__wbc,
    bd.neutrophils AS value__lab__neutrophil,
    co.d_dimer AS value__lab__d_dimer,
    co.fibrinogen AS value__lab__fibrinogen,
    co.inr AS value__lab__inr,
    co.pt AS value__lab__pt,
    co.ptt AS value__lab__aptt,
    en.bilirubin_total AS value__lab__bilirubin,
    inf.crp AS value__lab__crp,
    bg.baseexcess AS value__lab__base_excess,
    bg.lactate AS value__lab__lactate,
    bg.so2 AS value__lab__oxygen_saturation,
    bg.pao2fio2ratio AS value__lab__oxygenation_index,
    bg.pco2 AS value__lab__paco2,
    bg.po2 AS value__lab__pao2,
    bg.ph AS value__lab__ph,
    NULL::double precision AS value__lab__procalcitonin
FROM seppred_candidates c
LEFT JOIN LATERAL (SELECT * FROM seppred_vitalsign x WHERE x.stay_id=c.stay_id AND x.charttime BETWEEN c.prediction_time-interval '24 hour' AND c.prediction_time ORDER BY x.charttime DESC LIMIT 1) v ON true
LEFT JOIN LATERAL (SELECT * FROM mimiciv_derived.chemistry x WHERE x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id AND x.charttime BETWEEN c.prediction_time-interval '24 hour' AND c.prediction_time ORDER BY x.charttime DESC LIMIT 1) ch ON true
LEFT JOIN LATERAL (SELECT * FROM mimiciv_derived.complete_blood_count x WHERE x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id AND x.charttime BETWEEN c.prediction_time-interval '24 hour' AND c.prediction_time ORDER BY x.charttime DESC LIMIT 1) cb ON true
LEFT JOIN LATERAL (SELECT * FROM mimiciv_derived.blood_differential x WHERE x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id AND x.charttime BETWEEN c.prediction_time-interval '24 hour' AND c.prediction_time ORDER BY x.charttime DESC LIMIT 1) bd ON true
LEFT JOIN LATERAL (SELECT * FROM mimiciv_derived.coagulation x WHERE x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id AND x.charttime BETWEEN c.prediction_time-interval '24 hour' AND c.prediction_time ORDER BY x.charttime DESC LIMIT 1) co ON true
LEFT JOIN LATERAL (SELECT * FROM mimiciv_derived.enzyme x WHERE x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id AND x.charttime BETWEEN c.prediction_time-interval '24 hour' AND c.prediction_time ORDER BY x.charttime DESC LIMIT 1) en ON true
LEFT JOIN LATERAL (SELECT * FROM mimiciv_derived.inflammation x WHERE x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id AND x.charttime BETWEEN c.prediction_time-interval '24 hour' AND c.prediction_time ORDER BY x.charttime DESC LIMIT 1) inf ON true
LEFT JOIN LATERAL (SELECT * FROM mimiciv_derived.bg x WHERE x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id AND x.charttime BETWEEN c.prediction_time-interval '24 hour' AND c.prediction_time ORDER BY x.charttime DESC LIMIT 1) bg ON true
), with_masks AS (
    SELECT e.*,
      (value__lab__albumin IS NOT NULL)::integer AS mask__value__lab__albumin,
      (value__lab__hco3 IS NOT NULL)::integer AS mask__value__lab__hco3,
      (value__lab__urea IS NOT NULL)::integer AS mask__value__lab__urea,
      (value__lab__calcium IS NOT NULL)::integer AS mask__value__lab__calcium,
      (value__lab__chloride IS NOT NULL)::integer AS mask__value__lab__chloride,
      (value__lab__creatinine IS NOT NULL)::integer AS mask__value__lab__creatinine,
      (value__lab__glucose IS NOT NULL)::integer AS mask__value__lab__glucose,
      (value__lab__sodium IS NOT NULL)::integer AS mask__value__lab__sodium,
      (value__lab__potassium IS NOT NULL)::integer AS mask__value__lab__potassium,
      (value__lab__hemoglobin IS NOT NULL)::integer AS mask__value__lab__hemoglobin,
      (value__lab__platelet IS NOT NULL)::integer AS mask__value__lab__platelet,
      (value__lab__wbc IS NOT NULL)::integer AS mask__value__lab__wbc,
      (value__lab__neutrophil IS NOT NULL)::integer AS mask__value__lab__neutrophil,
      (value__lab__d_dimer IS NOT NULL)::integer AS mask__value__lab__d_dimer,
      (value__lab__fibrinogen IS NOT NULL)::integer AS mask__value__lab__fibrinogen,
      (value__lab__inr IS NOT NULL)::integer AS mask__value__lab__inr,
      (value__lab__pt IS NOT NULL)::integer AS mask__value__lab__pt,
      (value__lab__aptt IS NOT NULL)::integer AS mask__value__lab__aptt,
      (value__lab__bilirubin IS NOT NULL)::integer AS mask__value__lab__bilirubin,
      (value__lab__crp IS NOT NULL)::integer AS mask__value__lab__crp,
      (value__lab__base_excess IS NOT NULL)::integer AS mask__value__lab__base_excess,
      (value__lab__lactate IS NOT NULL)::integer AS mask__value__lab__lactate,
      (value__lab__oxygen_saturation IS NOT NULL)::integer AS mask__value__lab__oxygen_saturation,
      (value__lab__oxygenation_index IS NOT NULL)::integer AS mask__value__lab__oxygenation_index,
      (value__lab__paco2 IS NOT NULL)::integer AS mask__value__lab__paco2,
      (value__lab__pao2 IS NOT NULL)::integer AS mask__value__lab__pao2,
      (value__lab__ph IS NOT NULL)::integer AS mask__value__lab__ph,
      0::integer AS mask__value__lab__procalcitonin
    FROM endpoint e
)
SELECT * FROM with_masks ORDER BY horizon_h, label DESC, stay_id;

\copy seppred_endpoint TO 'ProcessedData/mimic_external_validation/mimic_endpoint_all.csv' WITH (FORMAT csv, HEADER true)

\echo 'Created mimic_endpoint_all.csv'
