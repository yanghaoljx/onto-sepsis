\set ON_ERROR_STOP on

-- Read-only source extraction for temporal external validation. All database
-- objects are session-local temporary tables and disappear on disconnect.
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
           COALESCE(s.sepsis3, FALSE) AS label
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
SELECT * FROM positive
UNION ALL
SELECT * FROM negative;

CREATE INDEX ON seppred_candidates(stay_id, prediction_time);
CREATE INDEX ON seppred_candidates(subject_id, hadm_id, prediction_time);

\copy (SELECT subject_id, hadm_id, stay_id, horizon_h, label::integer AS label, admission_age, icu_intime, prediction_time, sepsis_time FROM seppred_candidates ORDER BY horizon_h, label DESC, stay_id) TO 'ProcessedData/mimic_external_validation/mimic_sequence_candidates.csv' WITH (FORMAT csv, HEADER true)

-- vitalsign lacks a suitable time index in this installation. Restrict it to
-- relevant stays once and index the session-local copy.
CREATE TEMP TABLE seppred_vitalsign AS
SELECT v.*
FROM mimiciv_derived.vitalsign v
JOIN (SELECT DISTINCT stay_id FROM seppred_candidates) c USING (stay_id);
CREATE INDEX ON seppred_vitalsign(stay_id, charttime);

-- Emit the complete ICU history up to the prediction cutoff.  The local model
-- uses at most 24 ward-round snapshots (not a 24-hour lookback); Python turns
-- these source events into admission-day snapshots with cumulative masks.
CREATE TEMP TABLE seppred_events AS
SELECT c.subject_id, c.hadm_id, c.stay_id, c.horizon_h, c.label::integer AS label,
       c.admission_age, c.icu_intime, c.prediction_time,
       v.charttime AS event_time, 'vitalsign'::text AS source,
       NULL::double precision AS value__lab__albumin,
       NULL::double precision AS value__lab__aptt,
       NULL::double precision AS value__lab__base_excess,
       NULL::double precision AS value__lab__bilirubin,
       NULL::double precision AS value__lab__calcium,
       NULL::double precision AS value__lab__chloride,
       NULL::double precision AS value__lab__creatinine,
       NULL::double precision AS value__lab__crp,
       NULL::double precision AS value__lab__d_dimer,
       NULL::double precision AS value__lab__fibrinogen,
       NULL::double precision AS value__lab__glucose,
       NULL::double precision AS value__lab__hco3,
       NULL::double precision AS value__lab__hemoglobin,
       NULL::double precision AS value__lab__inr,
       NULL::double precision AS value__lab__lactate,
       NULL::double precision AS value__lab__neutrophil,
       NULL::double precision AS value__lab__oxygen_saturation,
       NULL::double precision AS value__lab__oxygenation_index,
       NULL::double precision AS value__lab__paco2,
       NULL::double precision AS value__lab__pao2,
       NULL::double precision AS value__lab__ph,
       NULL::double precision AS value__lab__platelet,
       NULL::double precision AS value__lab__potassium,
       NULL::double precision AS value__lab__procalcitonin,
       NULL::double precision AS value__lab__pt,
       NULL::double precision AS value__lab__sodium,
       NULL::double precision AS value__lab__urea,
       NULL::double precision AS value__lab__wbc,
       v.dbp::double precision AS value__vital__diastolic_bp,
       v.glucose::double precision AS value__vital__glucose,
       v.heart_rate::double precision AS value__vital__heart_rate,
       v.mbp::double precision AS value__vital__mean_bp,
       v.spo2::double precision AS value__vital__oxygen_saturation,
       v.resp_rate::double precision AS value__vital__respiratory_rate,
       v.sbp::double precision AS value__vital__systolic_bp,
       v.temperature::double precision AS value__vital__temperature
FROM seppred_candidates c
JOIN seppred_vitalsign v ON v.stay_id=c.stay_id
 AND v.charttime >= c.icu_intime
 AND v.charttime <= c.prediction_time

UNION ALL

SELECT c.subject_id, c.hadm_id, c.stay_id, c.horizon_h, c.label::integer,
       c.admission_age, c.icu_intime, c.prediction_time,
       x.charttime, 'chemistry',
       x.albumin, NULL, NULL, NULL, x.calcium, x.chloride, x.creatinine,
       NULL, NULL, NULL, x.glucose, x.bicarbonate, NULL, NULL, NULL, NULL,
       NULL, NULL, NULL, NULL, NULL, NULL, x.potassium, NULL, NULL, x.sodium,
       x.bun * 0.357, NULL,
       NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
FROM seppred_candidates c
JOIN mimiciv_derived.chemistry x ON x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id
 AND x.charttime >= c.icu_intime AND x.charttime <= c.prediction_time

UNION ALL

SELECT c.subject_id, c.hadm_id, c.stay_id, c.horizon_h, c.label::integer,
       c.admission_age, c.icu_intime, c.prediction_time,
       x.charttime, 'complete_blood_count',
       NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
       x.hemoglobin,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,x.platelet,
       NULL,NULL,NULL,NULL,NULL,x.wbc,
       NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL
FROM seppred_candidates c
JOIN mimiciv_derived.complete_blood_count x ON x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id
 AND x.charttime >= c.icu_intime AND x.charttime <= c.prediction_time

UNION ALL

SELECT c.subject_id, c.hadm_id, c.stay_id, c.horizon_h, c.label::integer,
       c.admission_age, c.icu_intime, c.prediction_time,
       x.charttime, 'blood_differential',
       NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
       x.neutrophils,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
       NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL
FROM seppred_candidates c
JOIN mimiciv_derived.blood_differential x ON x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id
 AND x.charttime >= c.icu_intime AND x.charttime <= c.prediction_time

UNION ALL

SELECT c.subject_id, c.hadm_id, c.stay_id, c.horizon_h, c.label::integer,
       c.admission_age, c.icu_intime, c.prediction_time,
       x.charttime, 'coagulation',
       NULL,x.ptt,NULL,NULL,NULL,NULL,NULL,NULL,x.d_dimer,x.fibrinogen,NULL,NULL,
       NULL,x.inr,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,x.pt,NULL,NULL,NULL,
       NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL
FROM seppred_candidates c
JOIN mimiciv_derived.coagulation x ON x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id
 AND x.charttime >= c.icu_intime AND x.charttime <= c.prediction_time

UNION ALL

SELECT c.subject_id, c.hadm_id, c.stay_id, c.horizon_h, c.label::integer,
       c.admission_age, c.icu_intime, c.prediction_time,
       x.charttime, 'enzyme',
       NULL,NULL,NULL,x.bilirubin_total,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
       NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
       NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL
FROM seppred_candidates c
JOIN mimiciv_derived.enzyme x ON x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id
 AND x.charttime >= c.icu_intime AND x.charttime <= c.prediction_time

UNION ALL

SELECT c.subject_id, c.hadm_id, c.stay_id, c.horizon_h, c.label::integer,
       c.admission_age, c.icu_intime, c.prediction_time,
       x.charttime, 'inflammation',
       NULL,NULL,NULL,NULL,NULL,NULL,NULL,x.crp,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
       NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
       NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL
FROM seppred_candidates c
JOIN mimiciv_derived.inflammation x ON x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id
 AND x.charttime >= c.icu_intime AND x.charttime <= c.prediction_time

UNION ALL

SELECT c.subject_id, c.hadm_id, c.stay_id, c.horizon_h, c.label::integer,
       c.admission_age, c.icu_intime, c.prediction_time,
       x.charttime, 'blood_gas',
       NULL,NULL,x.baseexcess,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
       x.lactate,NULL,x.so2,x.pao2fio2ratio,x.pco2,x.po2,x.ph,NULL,NULL,NULL,NULL,
       NULL,NULL,NULL,
       NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL
FROM seppred_candidates c
JOIN mimiciv_derived.bg x ON x.subject_id=c.subject_id AND x.hadm_id=c.hadm_id
 AND x.charttime >= c.icu_intime AND x.charttime <= c.prediction_time

;

\copy (SELECT * FROM seppred_events ORDER BY horizon_h, stay_id, event_time, source) TO 'ProcessedData/mimic_external_validation/mimic_sequence_events.csv' WITH (FORMAT csv, HEADER true)

\echo 'Created mimic_sequence_candidates.csv and mimic_sequence_events.csv'
