"""Build matched ward-round timepoint datasets for sepsis prediction.

Run build_sepsis_events.py first. This script creates one labeled sample per
ward-round timepoint, matched negative timepoints, and patient-level splits.
"""

import hashlib
from pathlib import Path

import polars as pl


COHORT_FILE = Path("ProcessedData/sepsis_cohort.csv")
BASE_FILE = Path("Dataset/baseinfo.csv")
WARD_FILE = Path("Dataset/icu_ward_rounds_HID0101.csv")
VITAL_FILE = Path("Dataset/icu_vital_signs_HID0101.csv")
LAB_FILE = Path("Dataset/lab_reportdet_HID0101.csv")
OUTPUT_DIR = Path("ProcessedData/model_dataset")

TIME_FMT = "%Y-%m-%d %H:%M:%S"
SPLIT_SEED = "sepsis_prediction_v1"
NEGATIVE_PATIENT_RATIO = 1.0


def parse_time(column):
    return column.str.strptime(
        pl.Datetime,
        TIME_FMT,
        strict=False,
    )


def read_cohort():
    if not COHORT_FILE.exists():
        raise SystemExit(
            "找不到 ProcessedData/sepsis_cohort.csv。"
            "请先运行 build_sepsis_events.py。"
        )

    return (
        pl.read_csv(
            COHORT_FILE,
            encoding="utf8-lossy",
            infer_schema_length=1000,
        )
        .select([
            pl.col("patient_id").cast(pl.String),
            pl.col("visit_no").cast(pl.String),
            pl.col("cohort").cast(pl.String),
            pl.col("label").cast(pl.String),
            parse_time(pl.col("event_time")).alias("event_time"),
            pl.col("event_type").cast(pl.String),
            pl.col("event_desc").cast(pl.String),
        ])
        .filter(pl.col("cohort").is_in(["positive", "negative"]))
    )


def read_base():
    return (
        pl.read_csv(
            BASE_FILE,
            encoding="utf8-lossy",
            infer_schema_length=1000,
        )
        .select([
            pl.col("就诊号").cast(pl.String).alias("visit_no"),
            pl.col("入院日期"),
            pl.col("入院时间"),
            pl.col("出院日期"),
            pl.col("出院时间"),
        ])
        .with_columns(
            (
                pl.col("入院日期")
                + pl.lit(" ")
                + pl.col("入院时间")
            )
            .str.strptime(
                pl.Datetime,
                TIME_FMT,
                strict=False,
            )
            .alias("admission_time")
        )
        .with_columns(
            (
                pl.col("出院日期")
                + pl.lit(" ")
                + pl.col("出院时间")
            )
            .str.strptime(
                pl.Datetime,
                TIME_FMT,
                strict=False,
            )
            .alias("discharge_time")
        )
        .select(["visit_no", "admission_time", "discharge_time"])
        .unique(subset=["visit_no"], keep="last")
    )


def read_ward_rounds():
    return (
        pl.scan_csv(
            WARD_FILE,
            encoding="utf8-lossy",
            ignore_errors=True,
            infer_schema_length=1000,
        )
        .select([
            pl.col("case_cdadet_visitno")
            .cast(pl.String)
            .alias("visit_no"),
            parse_time(
                pl.col("case_cdadet_createdttm")
            ).alias("round_time"),
            pl.col("case_cdadet_templatename")
            .alias("round_template"),
            pl.col("case_cdadet_datasetcode")
            .alias("round_code"),
            pl.col("case_cdadet_datasetvalue")
            .alias("round_text"),
            pl.col("case_cdadet_nursingdetstatusname")
            .alias("record_status"),
        ])
        .filter(
            (pl.col("round_code") == "S001")
            & (pl.col("record_status") != "删除")
            & pl.col("round_time").is_not_null()
            & pl.col("round_text").is_not_null()
            & (pl.col("round_text").str.len_chars() > 0)
        )
        .collect(engine="streaming")
    )


def add_time_fields(data):
    return data.with_columns([
        (
            (pl.col("round_time") - pl.col("admission_time"))
            .dt.total_seconds()
            / 3600
        ).alias("hours_from_admission"),
        pl.when(pl.col("cohort") == "positive")
        .then(
            (
                (pl.col("event_time") - pl.col("round_time"))
                .dt.total_seconds()
                / 3600
            )
        )
        .otherwise(None)
        .alias("hours_to_sepsis"),
    ])


def add_time_bins(data):
    return data.with_columns(
        pl.when(pl.col("hours_from_admission") < 6)
        .then(pl.lit("0_6h"))
        .when(pl.col("hours_from_admission") < 24)
        .then(pl.lit("6_24h"))
        .when(pl.col("hours_from_admission") < 48)
        .then(pl.lit("24_48h"))
        .when(pl.col("hours_from_admission") < 72)
        .then(pl.lit("48_72h"))
        .when(pl.col("hours_from_admission") < 168)
        .then(pl.lit("3_7d"))
        .otherwise(pl.lit("7d_plus"))
        .alias("time_bin")
    )


def add_patient_splits(data):
    patients = data.select("patient_id").unique()
    split_rows = []

    for patient_id in patients.get_column("patient_id").to_list():
        token = f"{SPLIT_SEED}:{patient_id}".encode()
        bucket = int(hashlib.sha1(token).hexdigest()[:8], 16) % 100

        if bucket < 70:
            split = "train"
        elif bucket < 85:
            split = "validation"
        else:
            split = "test"

        split_rows.append({
            "patient_id": patient_id,
            "split": split,
        })

    return data.join(
        pl.DataFrame(split_rows),
        on="patient_id",
        how="left",
    )


def build_candidates(cohort, base, ward):
    data = (
        ward
        .join(base, on="visit_no", how="inner")
        .join(cohort, on="visit_no", how="inner")
        .filter(
            (pl.col("round_time") >= pl.col("admission_time"))
            & (pl.col("round_time") <= pl.col("discharge_time"))
            & (
                (pl.col("cohort") == "negative")
                | (
                    (pl.col("cohort") == "positive")
                    & (pl.col("round_time") < pl.col("event_time"))
                )
            )
        )
        .drop(["round_code", "record_status"])
    )

    return add_time_fields(data)


def representative_time_bins(data):
    """Give each visit the most common admission-relative time bin."""
    return (
        add_time_bins(data)
        .group_by(["split", "patient_id", "visit_no", "time_bin"])
        .len()
        .sort("len", descending=True)
        .unique(subset=["split", "patient_id", "visit_no"], keep="first")
        .select(["split", "patient_id", "visit_no", "time_bin"])
    )


def select_negative_visits(candidates):
    """Select negative visits by positive time-bin proportions.

    Selection is at visit level. All timepoints from a selected negative
    visit are retained later; this is not one-to-one timepoint matching.
    """
    positive = candidates.filter(pl.col("cohort") == "positive")
    negative = candidates.filter(pl.col("cohort") == "negative")
    positive_profile = representative_time_bins(positive)
    negative_profile = representative_time_bins(negative)
    all_selected = []

    for split in ["train", "validation", "test"]:
        target = positive_profile.filter(pl.col("split") == split)
        pool = negative_profile.filter(pl.col("split") == split)
        target_n = round(target.height * NEGATIVE_PATIENT_RATIO)

        if target_n == 0 or pool.is_empty():
            continue

        # Keep the same proportions of representative time bins as positives.
        selected_for_split = []
        for time_bin_key, group in target.group_by("time_bin", maintain_order=True):
            time_bin = time_bin_key[0] if isinstance(time_bin_key, tuple) else time_bin_key
            requested = round(group.height / target.height * target_n)
            candidates_in_bin = pool.filter(pl.col("time_bin") == time_bin)
            if requested:
                seed = int(hashlib.sha1(
                    f"{SPLIT_SEED}:{split}:{time_bin}".encode()
                ).hexdigest()[:8], 16)
                selected_for_split.append(
                    candidates_in_bin.sample(
                        n=min(requested, candidates_in_bin.height),
                        with_replacement=False,
                        shuffle=True,
                        seed=seed,
                    )
                )

        chosen = (
            pl.concat(selected_for_split, how="vertical_relaxed")
            .unique(subset=["split", "patient_id", "visit_no"])
            if selected_for_split else pl.DataFrame()
        )

        # Fill any shortage from the remaining negative visits in this split.
        shortage = target_n - chosen.height
        if shortage > 0:
            remaining = pool.join(
                chosen.select(["split", "patient_id", "visit_no"]),
                on=["split", "patient_id", "visit_no"],
                how="anti",
            )
            seed = int(hashlib.sha1(
                f"{SPLIT_SEED}:{split}:fill".encode()
            ).hexdigest()[:8], 16)
            chosen = pl.concat([
                chosen,
                remaining.sample(
                    n=min(shortage, remaining.height),
                    with_replacement=False,
                    shuffle=True,
                    seed=seed,
                ),
            ], how="vertical_relaxed")

        all_selected.append(chosen)

    if not all_selected:
        return negative.head(0)

    selected_visits = pl.concat(all_selected, how="vertical_relaxed").select(
        ["split", "patient_id", "visit_no"]
    )
    return negative.join(
        selected_visits,
        on=["split", "patient_id", "visit_no"],
        how="inner",
    )


def make_labeled_samples(candidates):
    positive = candidates.filter(pl.col("cohort") == "positive").select([
        "patient_id", "visit_no", "round_time", "round_template", "round_text",
        "hours_from_admission", "split", "hours_to_sepsis",
    ]).with_columns([
        pl.lit(1).alias("label"),
        pl.lit("positive").alias("sample_type"),
    ])

    negative = select_negative_visits(candidates).select([
        "patient_id", "visit_no", "round_time", "round_template", "round_text",
        "hours_from_admission", "split",
    ]).with_columns([
        pl.lit(None, dtype=pl.Float64).alias("hours_to_sepsis"),
        pl.lit(0).alias("label"),
        pl.lit("negative").alias("sample_type"),
    ])

    return pl.concat([positive, negative], how="vertical_relaxed")


def load_measurements(path, kind, visit_ids, base):
    if kind == "vital":
        columns = [
            pl.col("case_icudet_visitno").cast(pl.String).alias("visit_no"),
            parse_time(
                pl.col("case_icudet_datacreatedttm")
            ).alias("record_time"),
            pl.col("case_icudet_datasetcode").alias("code"),
            pl.col("case_icudet_datasetname").alias("name"),
            pl.col("case_icudet_datasetvalue").alias("value"),
            pl.col("case_icudet_datasetvaluecode").alias("unit"),
        ]
    else:
        columns = [
            pl.col("mt_reportdet_visitno").cast(pl.String).alias("visit_no"),
            parse_time(
                pl.col("mt_reportdet_labdttm")
            ).alias("record_time"),
            pl.col("mt_reportdet_itemname").alias("code"),
            pl.col("mt_reportdet_sitemcnname").alias("name"),
            pl.col("mt_reportdet_labvalue").alias("value"),
            pl.col("mt_reportdet_unit").alias("unit"),
            pl.col("mt_reportdet_resultproperty").alias("result"),
        ]

    measurements = (
        pl.scan_csv(
            path,
            encoding="utf8-lossy",
            ignore_errors=True,
            infer_schema_length=1000,
        )
        .select(columns)
        .filter(
            pl.col("visit_no").is_in(visit_ids)
            & pl.col("record_time").is_not_null()
        )
        .collect(engine="streaming")
    )

    return (
        measurements
        .join(
            base.select(["visit_no", "admission_time", "discharge_time"]),
            on="visit_no",
            how="inner",
        )
        .filter(
            (pl.col("record_time") >= pl.col("admission_time"))
            & (pl.col("record_time") <= pl.col("discharge_time"))
        )
        .drop(["admission_time", "discharge_time"])
    )


def attach_latest(samples, source, kind, base):
    anchor = samples.select([
        "patient_id",
        "visit_no",
        "round_time",
    ]).with_row_index("sample_id")

    visit_ids = anchor.get_column("visit_no").unique().to_list()
    measurements = load_measurements(source, kind, visit_ids, base)

    if measurements.is_empty():
        return samples.with_columns(
            pl.lit(None).alias(f"latest_{kind}")
        )

    latest_time = (
        anchor
        .sort(["visit_no", "round_time"])
        .join_asof(
            measurements
            .select(["visit_no", "record_time"])
            .unique()
            .sort(["visit_no", "record_time"]),
            left_on="round_time",
            right_on="record_time",
            by="visit_no",
            strategy="backward",
            allow_exact_matches=False,
        )
        .select(["sample_id", "visit_no", "record_time"])
        .rename({"record_time": "latest_time"})
    )

    attached = (
        latest_time
        .join(
            measurements,
            left_on=["visit_no", "latest_time"],
            right_on=["visit_no", "record_time"],
            how="left",
        )
        .group_by("sample_id")
        .agg(
            pl.struct([
                "latest_time",
                "code",
                "name",
                "value",
                "unit",
                *(["result"] if kind == "lab" else []),
            ]).alias(f"latest_{kind}")
        )
    )

    return (
        samples.with_row_index("sample_id")
        .join(attached, on="sample_id", how="left")
        .drop("sample_id")
    )


def main():
    cohort = read_cohort()
    base = read_base()
    ward = read_ward_rounds()

    candidates = build_candidates(cohort, base, ward)
    candidates = add_patient_splits(candidates)
    samples = make_labeled_samples(candidates)
    base_with_discharge = (
        pl.read_csv(
            BASE_FILE,
            encoding="utf8-lossy",
            infer_schema_length=1000,
        )
        .select([
            pl.col("就诊号").cast(pl.String).alias("visit_no"),
            pl.col("入院日期"),
            pl.col("入院时间"),
            pl.col("出院日期"),
            pl.col("出院时间"),
        ])
        .with_columns([
            (pl.col("入院日期") + pl.lit(" ") + pl.col("入院时间"))
            .str.strptime(pl.Datetime, TIME_FMT, strict=False)
            .alias("admission_time"),
            (pl.col("出院日期") + pl.lit(" ") + pl.col("出院时间"))
            .str.strptime(pl.Datetime, TIME_FMT, strict=False)
            .alias("discharge_time"),
        ])
        .select(["visit_no", "admission_time", "discharge_time"])
        .unique(subset=["visit_no"], keep="last")
    )

    samples = attach_latest(samples, VITAL_FILE, "vital", base_with_discharge)
    samples = attach_latest(samples, LAB_FILE, "lab", base_with_discharge)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples.write_parquet(OUTPUT_DIR / "timepoint_dataset.parquet")

    for split in ["train", "validation", "test"]:
        samples.filter(pl.col("split") == split).write_parquet(
            OUTPUT_DIR / f"{split}.parquet"
        )

    print("总时间点样本:", samples.height)
    print(samples.group_by(["split", "label"]).len().sort(["split", "label"]))
    print(
        "纳入的阴性就诊数:",
        samples.filter(pl.col("label") == 0)
        .select(["patient_id", "visit_no"])
        .unique()
        .height,
    )
    print("输出目录:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
