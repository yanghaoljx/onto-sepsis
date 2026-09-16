"""Data preparation for irregularly sampled sepsis prediction experiments.

The source files contain event-time rows rather than a regular time grid.  This
module creates one prediction endpoint per admission and builds a prefix
sequence containing only observations at or before that endpoint.

The endpoint-per-admission rule is deliberate: it prevents a visit with many
rounds from dominating the loss and avoids evaluating several highly
correlated prefixes from the same admission.  The existing patient-level
train/validation/test split is preserved.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl


SPLITS = ("train", "validation", "test")
TASKS = ("pre_6h", "pre_12h", "pre_24h")


def _normalise_time(value: Any) -> str:
    """Return a stable key for Polars datetimes and JSON timestamps."""

    if value is None:
        return ""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, datetime.min.time())
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            # Keep non-ISO strings usable as a last-resort key.
            return text
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat(timespec="seconds")


def row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row.get("visit_no", "")), _normalise_time(row.get("round_time")))


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


@dataclass
class SequenceExample:
    x: np.ndarray
    delta_hours: np.ndarray
    hours_from_admission: np.ndarray
    observed_mask: np.ndarray
    label: int
    split: str
    patient_id: str
    visit_no: str
    endpoint_time: str
    available_mask: np.ndarray | None = None

    @property
    def length(self) -> int:
        return int(self.x.shape[0])


@dataclass
class SequenceBundle:
    task: str
    variant: str
    feature_names: list[str]
    examples: list[SequenceExample]
    endpoints: list[dict[str, Any]]

    def by_split(self, split: str) -> list[SequenceExample]:
        return [example for example in self.examples if example.split == split]


class FeatureScaler:
    """Train-only standardisation for sequence features.

    Mask features are kept in their original 0/1 scale. For numeric
    measurements, ``mask=0`` can mean either a valid causally carried-forward
    value or a never-observed zero sentinel. Non-zero carried-forward values
    are retained. Only ``mask=0 and value=0`` is imputed with the training
    median. Scaling statistics use current observations plus valid carried-
    forward values. Other continuous features use all training event rows.
    """

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = list(feature_names)
        self.scale_indices = np.asarray(
            [i for i, name in enumerate(self.feature_names) if not name.startswith("mask__")],
            dtype=np.int64,
        )
        self.value_indices = np.asarray(
            [i for i, name in enumerate(self.feature_names) if name.startswith("value__")],
            dtype=np.int64,
        )
        self.mean = np.zeros(len(self.feature_names), dtype=np.float32)
        self.std = np.ones(len(self.feature_names), dtype=np.float32)
        self.median = np.zeros(len(self.feature_names), dtype=np.float32)
        self.constant_features = np.zeros(len(self.feature_names), dtype=bool)

    def fit(self, examples: Iterable[SequenceExample]) -> "FeatureScaler":
        dim = len(self.feature_names)
        total = np.zeros(dim, dtype=np.float64)
        total_sq = np.zeros(dim, dtype=np.float64)
        count = np.zeros(dim, dtype=np.float64)
        value_samples: dict[int, list[np.ndarray]] = {
            int(index): [] for index in self.value_indices
        }
        for example in examples:
            values = np.asarray(example.x, dtype=np.float64)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            valid = np.ones_like(values, dtype=np.float64)
            if self.value_indices.size:
                numeric_valid = (
                    (example.observed_mask[:, self.value_indices] > 0)
                    | (values[:, self.value_indices] != 0)
                )
                if getattr(example,"available_mask",None) is not None:
                    numeric_valid = example.available_mask[:,self.value_indices] > 0
                valid[:, self.value_indices] = numeric_valid
                for local_index, feature_index in enumerate(self.value_indices):
                    selected = values[numeric_valid[:, local_index], feature_index]
                    if selected.size:
                        value_samples[int(feature_index)].append(selected.copy())
            total += (values * valid).sum(axis=0)
            total_sq += (np.square(values) * valid).sum(axis=0)
            count += valid.sum(axis=0)
        if not np.any(count > 0):
            raise ValueError("No training sequence rows available for fitting the scaler")
        safe_count = np.maximum(count, 1.0)
        mean = total / safe_count
        variance = np.maximum(total_sq / safe_count - np.square(mean), 0.0)
        never_observed = count == 0
        # A constant / absent training channel has no learnable variation.
        # Dividing a novel external value by 1e-4 can saturate a recurrent net.
        # Freeze these continuous channels at their training representation.
        self.constant_features = (variance <= 1e-8) | never_observed
        self.constant_features[[i for i, n in enumerate(self.feature_names) if n.startswith("mask__")]] = False
        variance[self.constant_features] = 1.0
        mean[never_observed] = 0.0
        variance[never_observed] = 1.0
        self.mean = mean.astype(np.float32)
        self.std = np.sqrt(variance).astype(np.float32)
        for feature_index, chunks in value_samples.items():
            if chunks:
                self.median[feature_index] = np.float32(
                    np.median(np.concatenate(chunks))
                )
        self.mean[[i for i in range(dim) if i not in set(self.scale_indices.tolist())]] = 0.0
        self.std[[i for i in range(dim) if i not in set(self.scale_indices.tolist())]] = 1.0
        return self

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FeatureScaler":
        scaler = cls(state["feature_names"])
        for name in ("mean", "std", "median"):
            if name in state:
                setattr(scaler, name, np.asarray(state[name], dtype=np.float32))
        if any(a.shape != (len(scaler.feature_names),) for a in (scaler.mean, scaler.std, scaler.median)):
            raise ValueError("Scaler arrays do not match feature order")
        if not np.isfinite(scaler.std).all() or np.any(scaler.std <= 0):
            raise ValueError("Scaler standard deviations must be finite and positive")
        if "constant_features" in state:
            scaler.constant_features = np.asarray(state["constant_features"], dtype=bool)
        else:
            # Compatibility with saved scalers that used sqrt(max(var, 1e-8)).
            scaler.constant_features = (scaler.std <= 1.0001e-4)
            scaler.constant_features[[i for i, n in enumerate(scaler.feature_names) if n.startswith("mask__")]] = False
        return scaler

    def transform_array(self, values: np.ndarray, observed_mask: np.ndarray, available_mask: np.ndarray | None = None) -> np.ndarray:
        raw = np.asarray(values, dtype=np.float32)
        observed = np.asarray(observed_mask, dtype=np.float32)
        if raw.shape != observed.shape or raw.shape[-1] != len(self.feature_names):
            raise ValueError("Feature values, observation masks and scaler order must agree")
        if not np.isfinite(raw).all():
            raise ValueError("Non-finite raw feature passed to scaler")
        scaled = ((raw - self.mean) / self.std).astype(np.float32)
        if self.value_indices.size:
            never_observed = ((observed[..., self.value_indices] <= 0)
                              & (raw[..., self.value_indices] == 0))
            if available_mask is not None:
                if np.shape(available_mask)!=raw.shape: raise ValueError('Availability shape mismatch')
                never_observed = np.asarray(available_mask)[...,self.value_indices] <= 0
            imputed = ((self.median[self.value_indices] - self.mean[self.value_indices])
                       / self.std[self.value_indices])
            scaled[..., self.value_indices] = np.where(
                never_observed, imputed, scaled[..., self.value_indices])
        scaled[..., self.constant_features] = 0.0
        return scaled

    def transform(self, examples: Iterable[SequenceExample]) -> None:
        for example in examples:
            example.x = self.transform_array(example.x, example.observed_mask,getattr(example,'available_mask',None))

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "median": self.median.tolist(),
            "numeric_scaling": "carry_forward_preserved_never_observed_training_median",
            "constant_features": self.constant_features.tolist(),
            "constant_feature_policy": "freeze_at_training_representation",
        }


def _read_rows(path: Path, limit_rows: int | None = None) -> list[dict[str, Any]]:
    frame = pl.read_parquet(path)
    if limit_rows is not None:
        frame = frame.head(limit_rows)
    return frame.to_dicts()


def _feature_cache_path(jsonl_path: Path, cache_dir: Path) -> Path:
    return cache_dir / f"{jsonl_path.stem}.features.npz"


def _write_feature_cache(
    cache_path: Path,
    keys: list[tuple[str, str]],
    vectors: np.ndarray,
    feature_order: list[str],
    source_mtime_ns: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    visit_nos = np.asarray([key[0] for key in keys], dtype="U64")
    timestamps = np.asarray([key[1] for key in keys], dtype="U32")
    np.savez(
        cache_path,
        visit_no=visit_nos,
        round_time=timestamps,
        feature_vector=vectors.astype(np.float32),
        feature_order=np.asarray(feature_order, dtype="U128"),
        source_mtime_ns=np.asarray([source_mtime_ns], dtype=np.int64),
    )


def load_feature_table(
    jsonl_path: Path,
    schema_path: Path,
    cache_dir: Path,
    max_rows: int | None = None,
) -> tuple[dict[tuple[str, str], np.ndarray], list[str]]:
    """Load compact model features, creating a cache from the large JSONL once.

    The JSONL files contain large raw LLM responses.  The temporal models only
    need ``model_features.feature_vector``.  The compact cache prevents every
    model and every re-run from reparsing those large responses.
    """

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_order = list(schema["feature_order"])
    cache_path = _feature_cache_path(jsonl_path, cache_dir)
    source_mtime_ns = jsonl_path.stat().st_mtime_ns
    use_cache = (
        max_rows is None
        and cache_path.exists()
        and cache_path.stat().st_mtime_ns >= source_mtime_ns
    )

    if use_cache:
        cached = np.load(cache_path, allow_pickle=False)
        cached_order = [str(x) for x in cached["feature_order"].tolist()]
        if cached_order != feature_order:
            use_cache = False
        else:
            keys = zip(cached["visit_no"].tolist(), cached["round_time"].tolist())
            vectors = cached["feature_vector"]
            return {tuple(key): vectors[i] for i, key in enumerate(keys)}, feature_order

    keys: list[tuple[str, str]] = []
    vectors: list[np.ndarray] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_rows is not None and line_number > max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            features = row.get("model_features") or {}
            vector = features.get("feature_vector")
            if vector is None:
                raise ValueError(f"Missing model_features.feature_vector at {jsonl_path}:{line_number}")
            vector_array = np.asarray(vector, dtype=np.float32)
            if vector_array.size != len(feature_order):
                raise ValueError(
                    f"Feature dimension mismatch at {jsonl_path}:{line_number}: "
                    f"{vector_array.size} != {len(feature_order)}"
                )
            keys.append(row_key(row))
            vectors.append(vector_array)

    if not vectors:
        raise ValueError(f"No feature rows found in {jsonl_path}")
    matrix = np.stack(vectors).astype(np.float32, copy=False)
    if max_rows is None:
        _write_feature_cache(cache_path, keys, matrix, feature_order, source_mtime_ns)
    return {key: matrix[i] for i, key in enumerate(keys)}, feature_order


def _choose_positive_endpoint(rows: list[dict[str, Any]], anchor_hours: float) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if int(_as_float(row.get("label"))) == 1
        and (_as_optional_float(row.get("hours_to_sepsis")) is not None)
        and _as_float(row.get("hours_to_sepsis")) >= anchor_hours
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            abs(_as_float(row.get("hours_to_sepsis")) - anchor_hours),
            _normalise_time(row.get("round_time")),
        ),
    )


def _choose_negative_endpoint(
    rows: list[dict[str, Any]],
    target_hours: float,
) -> dict[str, Any]:
    candidates = [row for row in rows if int(_as_float(row.get("label"))) == 0]
    if not candidates:
        raise ValueError("Negative visit has no label=0 rows")
    return min(
        candidates,
        key=lambda row: (
            abs(_as_float(row.get("hours_from_admission")) - target_hours),
            _normalise_time(row.get("round_time")),
        ),
    )


def select_endpoints(rows: list[dict[str, Any]], anchor_hours: float) -> list[dict[str, Any]]:
    """Select one deterministic endpoint per visit.

    Positive endpoints are the observed rows closest to the requested
    pre-sepsis anchor.  Negative endpoints are pseudo-anchors matched to the
    *training* positive admission-relative endpoint-time distribution.  Using
    the training reference for every split prevents validation/test outcomes
    from influencing negative-index-time construction.
    """

    visits: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = str(row.get("split", ""))
        visit = str(row.get("visit_no", ""))
        if split in SPLITS and visit:
            visits[(split, visit)].append(row)

    positive_endpoints: dict[str, list[dict[str, Any]]] = defaultdict(list)
    negative_visits: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for (split, _), visit_rows in sorted(visits.items()):
        positive = _choose_positive_endpoint(visit_rows, anchor_hours)
        if positive is not None:
            positive_endpoints[split].append(positive)
        elif any(int(_as_float(row.get("label"))) == 0 for row in visit_rows):
            negative_visits[split].append(visit_rows)

    endpoints: list[dict[str, Any]] = []
    for split in SPLITS:
        positives = positive_endpoints[split]
        positives = sorted(positives, key=lambda row: _as_float(row.get("hours_from_admission")))
        for row in positives:
            endpoints.append({"row": row, "label": 1, "split": split})

        if positive_endpoints["train"]:
            reference_positives = sorted(
                positive_endpoints["train"],
                key=lambda row: _as_float(row.get("hours_from_admission")),
            )
        else:
            reference_positives = positives
        positive_times = [
            _as_float(row.get("hours_from_admission")) for row in reference_positives
        ]
        negative_groups = sorted(
            negative_visits[split],
            key=lambda group: _stable_hash(str(group[0].get("visit_no", ""))),
        )
        for index, visit_rows in enumerate(negative_groups):
            if positive_times:
                quantile_index = round(
                    index * (len(positive_times) - 1) / max(len(negative_groups) - 1, 1)
                )
                target = positive_times[quantile_index]
            else:
                target = 0.0
            negative = _choose_negative_endpoint(visit_rows, target)
            endpoints.append({"row": negative, "label": 0, "split": split})
    return endpoints


def _select_feature_indices(feature_order: Sequence[str], variant: str) -> list[int]:
    if variant not in {"structured", "state", "relation", "ontology"}:
        raise ValueError(f"Unknown feature variant: {variant}")
    if variant == "ontology":
        return list(range(len(feature_order)))
    # The extended ablation separates ontology-derived clinical states from
    # ontology relation counts.  ``structured`` is the pre-ontology baseline.
    return [
        i
        for i, name in enumerate(feature_order)
        if not (
            (variant == "structured" and (name.startswith("state__") or name.startswith("relation_count__")))
            or (variant == "state" and name.startswith("relation_count__"))
            or (variant == "relation" and name.startswith("state__"))
        )
    ]


def _mask_for_feature(name: str, vector: np.ndarray, feature_order: Sequence[str]) -> float:
    if name.startswith("value__"):
        mask_name = "mask__" + name
        try:
            index = feature_order.index(mask_name)
        except ValueError:
            return 1.0
        return float(vector[index] > 0.0)
    return 1.0


def _make_sequence(
    endpoint: dict[str, Any],
    rows_by_visit: Mapping[tuple[str, str], list[dict[str, Any]]],
    feature_table: Mapping[tuple[str, str], np.ndarray],
    feature_order: Sequence[str],
    feature_indices: Sequence[int],
    max_seq_len: int,
    available_table: Mapping | None = None,
) -> SequenceExample:
    endpoint_row = endpoint["row"]
    visit_key = (str(endpoint["split"]), str(endpoint_row.get("visit_no", "")))
    endpoint_time = _normalise_time(endpoint_row.get("round_time"))
    endpoint_dt = endpoint_time
    visit_rows = [
        row
        for row in rows_by_visit[visit_key]
        if _normalise_time(row.get("round_time")) <= endpoint_dt
    ]
    visit_rows.sort(key=lambda row: _normalise_time(row.get("round_time")))
    if not visit_rows:
        visit_rows = [endpoint_row]
    visit_rows = visit_rows[-max_seq_len:]

    raw_names = [feature_order[i] for i in feature_indices]
    vectors: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    admission_hours: list[float] = []
    available_rows = []
    for row in visit_rows:
        key = row_key(row)
        if key not in feature_table:
            raise KeyError(
                f"No ontology feature vector for visit_no={key[0]}, round_time={key[1]}"
            )
        full_vector = np.asarray(feature_table[key], dtype=np.float32)
        vectors.append(full_vector[list(feature_indices)])
        masks.append(
            np.asarray(
                [_mask_for_feature(name, full_vector, feature_order) for name in raw_names],
                dtype=np.float32,
            )
        )
        admission_hours.append(_as_float(row.get("hours_from_admission")))
        if available_table is not None:
            available_rows.append(np.asarray([float(n in available_table[key]) if n.startswith('value__lab__')
                else float(masks[-1][j]>0 or vectors[-1][j]!=0) if n.startswith('value__') else 1.
                for j,n in enumerate(raw_names)],np.float32))

    x = np.stack(vectors).astype(np.float32, copy=False)
    mask = np.stack(masks).astype(np.float32, copy=False)
    hours = np.asarray(admission_hours, dtype=np.float32)
    delta = np.diff(hours, prepend=hours[:1]).astype(np.float32)
    delta[0] = 0.0
    delta = np.maximum(delta, 0.0)
    temporal = np.stack([delta / 24.0, hours / 24.0], axis=1).astype(np.float32)
    x = np.concatenate([x, temporal], axis=1)
    mask = np.concatenate([mask, np.ones_like(temporal)], axis=1)

    return SequenceExample(
        x=np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        delta_hours=delta,
        hours_from_admission=hours,
        observed_mask=mask,
        label=int(endpoint["label"]),
        split=str(endpoint["split"]),
        patient_id=str(endpoint_row.get("patient_id", "")),
        visit_no=str(endpoint_row.get("visit_no", "")),
        endpoint_time=endpoint_time,
        available_mask=np.concatenate([np.stack(available_rows),np.ones_like(temporal)],axis=1) if available_rows else None,
    )


def build_bundle(
    task: str,
    variant: str,
    data_dir: Path,
    ontology_dir: Path,
    schema_path: Path,
    cache_dir: Path,
    max_seq_len: int = 24,
    max_rows: int | None = None,
    max_visits_per_split: int | None = None,
    repair_numeric_mappings: bool = True,
    numeric_contract: str = "v3",
    audit_path: Path | None = None,
) -> SequenceBundle:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    parquet_path = data_dir / f"{task}_all.parquet"
    if not parquet_path.exists():
        # Existing exports use separate split files; combine them in memory.
        paths = [data_dir / f"{task}_{split}.parquet" for split in SPLITS]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Cannot find {parquet_path} or split parquet files: {missing}"
            )
        split_frames = []
        for path in paths:
            split_frame = pl.read_parquet(path)
            if max_rows is not None:
                split_frame = split_frame.head(max_rows)
            split_frames.append(split_frame)
        frame = pl.concat(split_frames, how="vertical_relaxed")
        rows = frame.to_dicts()
    else:
        rows = _read_rows(parquet_path, max_rows)

    duplicate_rows_removed=0
    if numeric_contract in ('v4','v5'):
        # Repeated narrative templates can share the exact same clinical
        # snapshot. Keep one time step; reject clinically conflicting keys.
        unique={}
        clinical=('split','patient_id','label','hours_from_admission','hours_to_sepsis','latest_lab','latest_vital')
        for row in rows:
            key=row_key(row)
            if key in unique:
                if any(row.get(c)!=unique[key].get(c) for c in clinical):
                    raise ValueError('Conflicting clinical data at a duplicate visit/time key')
                duplicate_rows_removed+=1
            else:unique[key]=row
        rows=list(unique.values())

    jsonl_paths = [ontology_dir / f"{task}_{split}.jsonl" for split in SPLITS]
    missing_jsonl = [str(path) for path in jsonl_paths if not path.exists()]
    if missing_jsonl:
        raise FileNotFoundError(f"Missing ontology JSONL files: {missing_jsonl}")

    feature_table: dict[tuple[str, str], np.ndarray] = {}
    feature_order: list[str] | None = None
    for path in jsonl_paths:
        table, order = load_feature_table(path, schema_path, cache_dir, max_rows=max_rows)
        feature_table.update(table)
        if feature_order is None:
            feature_order = order
        elif feature_order != order:
            raise ValueError("Feature order differs across split JSONL files")
    assert feature_order is not None

    available_table = None
    if numeric_contract not in ("v3", "v4", "v5"):
        raise ValueError("Unknown numeric contract")
    if numeric_contract in ("v4","v5"):
        if variant != "structured":
            raise ValueError("Rebuild numeric structured input first, then derive the partial ontology")
        from clinical_feature_contract import rebuild_from_all_events
        event_path=data_dir.parent/('verified_lab_events_v4.parquet' if numeric_contract=='v4' else 'verified_numeric_events_v5.parquet')
        if not event_path.exists():raise FileNotFoundError('Run prepare_verified_lab_events.py before v4 training')
        audit,available_table = rebuild_from_all_events(rows,feature_table,feature_order,row_key,_normalise_time,event_path,include_vitals=numeric_contract=='v5')
        audit['identical_clinical_snapshot_duplicates_removed']=duplicate_rows_removed
        if audit_path is not None:
            audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    elif repair_numeric_mappings:
        # Overlay the original timestamped lab observations in memory. This
        # fixes stale cached/LLM vectors without rewriting the source JSONL.
        from numeric_feature_repairs import repair_feature_table
        repair_feature_table(rows,feature_table,feature_order,row_key,_normalise_time)

    endpoints = select_endpoints(rows, float(task.split("_")[1][:-1]))
    if max_visits_per_split is not None:
        selected: list[dict[str, Any]] = []
        for split in SPLITS:
            split_endpoints = [item for item in endpoints if item["split"] == split]
            split_endpoints.sort(
                key=lambda item: (str(item["row"].get("visit_no", "")), int(item["label"]))
            )
            positives = [item for item in split_endpoints if item["label"] == 1]
            negatives = [item for item in split_endpoints if item["label"] == 0]
            half = max_visits_per_split // 2
            chosen = positives[:half] + negatives[:half]
            if len(chosen) < max_visits_per_split:
                remaining = [item for item in split_endpoints if item not in chosen]
                chosen.extend(remaining[: max_visits_per_split - len(chosen)])
            selected.extend(chosen[:max_visits_per_split])
        endpoints = selected

    rows_by_visit: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_visit[(str(row.get("split", "")), str(row.get("visit_no", "")))].append(row)

    feature_indices = _select_feature_indices(feature_order, variant)
    if numeric_contract in ("v4","v5"):
        from clinical_feature_contract import EXCLUDED_FEATURES,EXCLUDED_FEATURES_V5
        excluded=EXCLUDED_FEATURES if numeric_contract=='v4' else EXCLUDED_FEATURES_V5
        feature_indices = [i for i in feature_indices if feature_order[i].removeprefix("mask__") not in excluded]
    base_names = [feature_order[i] for i in feature_indices]
    feature_names = base_names + ["time__delta_hours", "time__hours_from_admission"]
    examples = [
        _make_sequence(
            endpoint,
            rows_by_visit,
            feature_table,
            feature_order,
            feature_indices,
            max_seq_len,
            available_table,
        )
        for endpoint in endpoints
    ]
    return SequenceBundle(
        task=task,
        variant=variant,
        feature_names=feature_names,
        examples=examples,
        endpoints=endpoints,
    )


def endpoint_matrix(
    examples: Sequence[SequenceExample],
    pooling: str = "summary",
) -> np.ndarray:
    """Convert a variable-length sequence to a fixed endpoint representation.

    ``summary`` is the main tabular baseline and includes last/mean/min/max
    pooling.  ``last`` is an audit mode that tests whether a tree model's
    advantage is driven mainly by this engineered temporal pooling.
    """

    if pooling not in {"summary", "last"}:
        raise ValueError(f"Unknown tabular pooling: {pooling}")

    rows: list[np.ndarray] = []
    for example in examples:
        x = example.x
        last = x[-1]
        if pooling == "last":
            rows.append(last.astype(np.float32, copy=False))
            continue
        mean = x.mean(axis=0)
        minimum = x.min(axis=0)
        maximum = x.max(axis=0)
        summary = np.asarray(
            [
                float(example.length),
                float(example.hours_from_admission[-1] - example.hours_from_admission[0]),
                float(example.delta_hours[-1]),
            ],
            dtype=np.float32,
        )
        rows.append(np.concatenate([last, mean, minimum, maximum, summary]))
    return np.stack(rows).astype(np.float32)


def endpoint_feature_names(feature_names: Sequence[str], pooling: str = "summary") -> list[str]:
    if pooling == "last":
        return list(feature_names)
    if pooling != "summary":
        raise ValueError(f"Unknown tabular pooling: {pooling}")
    return (
        [f"last::{name}" for name in feature_names]
        + [f"mean::{name}" for name in feature_names]
        + [f"min::{name}" for name in feature_names]
        + [f"max::{name}" for name in feature_names]
        + ["sequence_length", "observed_duration_hours", "last_delta_hours"]
    )


def labels(examples: Sequence[SequenceExample]) -> np.ndarray:
    return np.asarray([example.label for example in examples], dtype=np.int64)


def metadata_rows(examples: Sequence[SequenceExample]) -> list[dict[str, Any]]:
    return [
        {
            "patient_id": example.patient_id,
            "visit_no": example.visit_no,
            "split": example.split,
            "label": example.label,
            "endpoint_time": example.endpoint_time,
            "sequence_length": example.length,
            "hours_from_admission": float(example.hours_from_admission[-1]),
        }
        for example in examples
    ]
