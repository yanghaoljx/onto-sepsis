"""Audit enhanced JSONL files for outcome-word and post-cutoff leakage."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


OUTCOME_WORDS = re.compile(
    r"脓毒症|败血症|脓毒性休克|感染性休克|脓毒症休克|"
    r"sepsis|septic\s+shock|DIC诊断|弥散性血管内凝血|"
    r"多器官功能障碍综合征|MODS",
    flags=re.I,
)


def _parse_time(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _iter_items(row: dict, key: str):
    value = row.get(key) or []
    if isinstance(value, list):
        yield from (item for item in value if isinstance(item, dict))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ontology-dir",
        type=Path,
        default=Path("ProcessedData/model_dataset/ontology_enhanced"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.ontology_dir / "leakage_audit.json"

    report = {
        "files": {},
        "total_rows": 0,
        "rows_with_excluded_outcome_mentions": 0,
        "rows_with_accepted_outcome_evidence": 0,
        "accepted_outcome_evidence_by_field": {},
        "accepted_outcome_evidence_examples": [],
        "post_cutoff_observations": 0,
        "post_cutoff_examples": [],
    }
    for path in sorted(args.ontology_dir.glob("pre_*_*.jsonl")):
        file_report = {
            "rows": 0,
            "excluded_outcome_mentions": 0,
            "accepted_outcome_evidence": 0,
            "post_cutoff_observations": 0,
        }
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                file_report["rows"] += 1
                report["total_rows"] += 1
                excluded = row.get("excluded_outcome_mentions") or []
                if excluded:
                    file_report["excluded_outcome_mentions"] += 1
                    report["rows_with_excluded_outcome_mentions"] += 1

                round_time = _parse_time(row.get("round_time"))
                accepted = []
                for key in ("structured_observations", "ward_round_observations"):
                    for item in _iter_items(row, key):
                        evidence = str(
                            item.get("evidence")
                            or item.get("source_text")
                            or item.get("raw_result")
                            or ""
                        )
                        if OUTCOME_WORDS.search(evidence):
                            accepted.append({"field": key, "concept_id": item.get("concept_id"), "evidence": evidence})
                        effective_time = _parse_time(item.get("effective_time"))
                        if round_time and effective_time and effective_time > round_time:
                            file_report["post_cutoff_observations"] += 1
                            report["post_cutoff_observations"] += 1
                            if len(report["post_cutoff_examples"]) < 20:
                                report["post_cutoff_examples"].append(
                                    {
                                        "file": str(path),
                                        "line": line_number,
                                        "round_time": row.get("round_time"),
                                        "effective_time": item.get("effective_time"),
                                        "concept_id": item.get("concept_id"),
                                    }
                                )
                for item in _iter_items(row, "relations"):
                    evidence = str(item.get("evidence") or item.get("source_text") or "")
                    subject = str(item.get("subject_concept") or item.get("subject") or "")
                    obj = str(item.get("object_concept") or item.get("object") or "")
                    if OUTCOME_WORDS.search(f"{evidence} {subject} {obj}"):
                        accepted.append({"field": "relations", "subject": subject, "object": obj, "evidence": evidence})
                if accepted:
                    file_report["accepted_outcome_evidence"] += 1
                    report["rows_with_accepted_outcome_evidence"] += 1
                    for item in accepted:
                        field = item.get("field", "unknown")
                        report["accepted_outcome_evidence_by_field"][field] = (
                            report["accepted_outcome_evidence_by_field"].get(field, 0) + 1
                        )
                    if len(report["accepted_outcome_evidence_examples"]) < 20:
                        report["accepted_outcome_evidence_examples"].append(
                            {
                                "file": str(path),
                                "line": line_number,
                                "visit_no": row.get("visit_no"),
                                "round_time": row.get("round_time"),
                                "label": row.get("label"),
                                "hours_to_sepsis": row.get("hours_to_sepsis"),
                                "prediction_group": row.get("prediction_group"),
                                "items": accepted,
                            }
                        )
        report["files"][str(path)] = file_report

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Audit written to {output}")


if __name__ == "__main__":
    main()
