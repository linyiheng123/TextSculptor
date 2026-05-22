#!/usr/bin/env python3
"""Summarize TextSculpt-Bench evaluation results across models and edit types."""

import argparse
import json
import os
import sys
from typing import Any, Optional

from metrics.utils import ensure_dir, load_jsonl_lines


EDIT_TYPE_ORDER = ["add", "remove", "replace", "hybrid"]


def as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_json_file(file_path: str) -> Optional[dict[str, Any]]:
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def count_existing_records(details_path: str, total_samples: Optional[int]) -> Optional[int]:
    if os.path.exists(details_path):
        return len(load_jsonl_lines(details_path))
    return total_samples


def load_instruction_counts(instruction_dir: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edit_type in EDIT_TYPE_ORDER:
        path = os.path.join(instruction_dir, f"{edit_type}.jsonl")
        if not os.path.exists(path):
            raise ValueError(f"Missing instruction file for {edit_type}: {path}")
        counts[edit_type] = len(load_jsonl_lines(path))
    return counts


def discover_models(eval_root: str) -> list[str]:
    models: list[str] = []
    if not os.path.isdir(eval_root):
        return models
    for name in sorted(os.listdir(eval_root)):
        full_path = os.path.join(eval_root, name)
        if os.path.isdir(full_path):
            models.append(name)
    return models


def compute_overall(text_accuracy: float, visual_quality: float, bp_preservation: float) -> float:
    return (text_accuracy + visual_quality + bp_preservation) / 3.0


def weighted_average_metric(values_with_weights: list[tuple[Optional[float], int]]) -> Optional[float]:
    weighted_sum = 0.0
    total_weight = 0
    for value, weight in values_with_weights:
        if value is None or weight <= 0:
            continue
        weighted_sum += value * weight
        total_weight += weight
    if total_weight == 0:
        return None
    return weighted_sum / total_weight


def load_bp_average(run_dir: str, bp_json: Optional[dict[str, Any]]) -> Optional[float]:
    bp_score = as_float(bp_json.get("average_bp_score") if bp_json else None)
    if bp_score is not None:
        return bp_score

    details_path = os.path.join(run_dir, "bp_details.jsonl")
    if not os.path.exists(details_path):
        return None

    records = load_jsonl_lines(details_path)
    ssim_values = [as_float(record.get("ssim")) for record in records if isinstance(record, dict)]
    valid_values = [value for value in ssim_values if value is not None]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)


def load_edit_type_summary(run_dir: str, expected_count: int) -> dict[str, Any]:
    ta_vq_json = load_json_file(os.path.join(run_dir, "ta_vq.json"))
    bp_json = load_json_file(os.path.join(run_dir, "bp.json"))

    ta_vq_total_samples = None
    if ta_vq_json and isinstance(ta_vq_json.get("total_samples"), int):
        ta_vq_total_samples = int(ta_vq_json["total_samples"])

    bp_total_samples = None
    if bp_json and isinstance(bp_json.get("total_samples"), int):
        bp_total_samples = int(bp_json["total_samples"])

    ta_vq_count = count_existing_records(os.path.join(run_dir, "ta_vq_details.jsonl"), ta_vq_total_samples)
    bp_count = count_existing_records(os.path.join(run_dir, "bp_details.jsonl"), bp_total_samples)

    ta_vq_complete = ta_vq_count == expected_count
    bp_complete = bp_count == expected_count
    complete = ta_vq_complete and bp_complete

    result = {
        "expected_count": expected_count,
        "ta_vq_count": ta_vq_count,
        "bp_count": bp_count,
        "ta_vq_complete": ta_vq_complete,
        "bp_complete": bp_complete,
        "complete": complete,
        "text_accuracy": None,
        "visual_quality": None,
        "bp_preservation": None,
        "overall": None,
    }
    if not complete:
        return result

    text_accuracy = as_float(ta_vq_json.get("average_text_accuracy_score") if ta_vq_json else None)
    visual_quality = as_float(ta_vq_json.get("average_visual_quality_score") if ta_vq_json else None)
    bp_preservation = load_bp_average(run_dir, bp_json)
    if text_accuracy is None or visual_quality is None or bp_preservation is None:
        return result

    result.update(
        {
            "text_accuracy": text_accuracy,
            "visual_quality": visual_quality,
            "bp_preservation": bp_preservation,
            "overall": compute_overall(text_accuracy, visual_quality, bp_preservation),
        }
    )
    return result


def build_model_summary(model: str, eval_root: str, instruction_counts: dict[str, int]) -> dict[str, Any]:
    per_type: dict[str, dict[str, Any]] = {}
    for edit_type in EDIT_TYPE_ORDER:
        run_dir = os.path.join(eval_root, model, edit_type)
        per_type[edit_type] = load_edit_type_summary(run_dir, instruction_counts[edit_type])

    overall_text_accuracy = weighted_average_metric(
        [(per_type[edit_type]["text_accuracy"], instruction_counts[edit_type]) for edit_type in EDIT_TYPE_ORDER]
    )
    overall_visual_quality = weighted_average_metric(
        [(per_type[edit_type]["visual_quality"], instruction_counts[edit_type]) for edit_type in EDIT_TYPE_ORDER]
    )
    overall_bp_preservation = weighted_average_metric(
        [(per_type[edit_type]["bp_preservation"], instruction_counts[edit_type]) for edit_type in EDIT_TYPE_ORDER]
    )
    overall_score = None
    if (
        overall_text_accuracy is not None
        and overall_visual_quality is not None
        and overall_bp_preservation is not None
    ):
        overall_score = compute_overall(overall_text_accuracy, overall_visual_quality, overall_bp_preservation)

    return {
        "model": model,
        "per_type": per_type,
        "overall": {
            "text_accuracy": overall_text_accuracy,
            "visual_quality": overall_visual_quality,
            "bp_preservation": overall_bp_preservation,
            "overall": overall_score,
        },
    }


def format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def build_completeness_rows(model_summaries: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for summary in model_summaries:
        model = summary["model"]
        for edit_type in EDIT_TYPE_ORDER:
            info = summary["per_type"][edit_type]
            rows.append(
                [
                    model,
                    edit_type,
                    format_value(info["expected_count"]),
                    format_value(info["ta_vq_count"]),
                    format_value(info["bp_count"]),
                    format_value(info["ta_vq_complete"]),
                    format_value(info["bp_complete"]),
                    format_value(info["complete"]),
                ]
            )
    return rows


def build_summary_rows(model_summaries: list[dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    headers = ["Model"]
    for edit_type in EDIT_TYPE_ORDER + ["overall"]:
        prefix = edit_type.capitalize()
        headers.extend([f"{prefix} TA", f"{prefix} VQ", f"{prefix} BP", f"{prefix} Overall"])

    rows: list[list[str]] = []
    for summary in model_summaries:
        row = [summary["model"]]
        for edit_type in EDIT_TYPE_ORDER:
            info = summary["per_type"][edit_type]
            row.extend(
                [
                    format_value(info["text_accuracy"]),
                    format_value(info["visual_quality"]),
                    format_value(info["bp_preservation"]),
                    format_value(info["overall"]),
                ]
            )

        overall = summary["overall"]
        row.extend(
            [
                format_value(overall["text_accuracy"]),
                format_value(overall["visual_quality"]),
                format_value(overall["bp_preservation"]),
                format_value(overall["overall"]),
            ]
        )
        rows.append(row)

    return headers, rows


def build_report(model_summaries: list[dict[str, Any]], eval_root: str, instruction_dir: str) -> str:
    lines = [
        "Evaluation Summary Report",
        f"- eval_root: {eval_root}",
        f"- instruction_dir: {instruction_dir}",
        "",
    ]

    completeness_headers = ["Model", "Type", "Expected", "TA/VQ", "BP", "TA/VQ Complete", "BP Complete", "Complete"]
    summary_headers, summary_rows = build_summary_rows(model_summaries)

    lines.append("[Completeness]")
    lines.append(make_table(completeness_headers, build_completeness_rows(model_summaries)))
    lines.append("")
    lines.append("[Summary]")
    lines.append(make_table(summary_headers, summary_rows))
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize model evaluation results across edit types")
    parser.add_argument(
        "--eval-root",
        default="benchmark/evaluations",
        help="Root directory containing per-model evaluation folders",
    )
    parser.add_argument(
        "--instruction-dir",
        default="benchmark/instructions",
        help="Directory containing {edit_type}.jsonl files",
    )
    parser.add_argument(
        "--output-txt",
        default="results/model_summary.txt",
        help="Path to output txt report",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isdir(args.eval_root):
        print(f"Error: Evaluation root not found: {args.eval_root}")
        sys.exit(1)
    if not os.path.isdir(args.instruction_dir):
        print(f"Error: Instruction directory not found: {args.instruction_dir}")
        sys.exit(1)

    try:
        instruction_counts = load_instruction_counts(args.instruction_dir)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    models = discover_models(args.eval_root)
    if not models:
        print(f"Error: No model directories found under {args.eval_root}")
        sys.exit(1)

    model_summaries = [build_model_summary(model, args.eval_root, instruction_counts) for model in models]
    report_text = build_report(model_summaries, args.eval_root, args.instruction_dir)

    output_path = args.output_txt
    ensure_dir(os.path.dirname(os.path.abspath(output_path)) or ".")
    try:
        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write(report_text)
    except OSError as exc:
        print(f"Error: Failed to write txt report: {exc}")
        sys.exit(1)

    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
