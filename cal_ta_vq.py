#!/usr/bin/env python3
"""Step 4: Calculate TA/VQ scores for edited images."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, cast

from tqdm import tqdm


@dataclass
class SimpleEditData:
    """Minimal edit payload needed by the TA/VQ metric."""

    edit_type: str
    instruction: str
    expected_edit_word_count: int = 1


def as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def as_int(value: object, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return {}


def get_existing_progress(filepath: str) -> set[str]:
    processed_files: set[str] = set()

    if not os.path.exists(filepath):
        return processed_files

    try:
        with open(filepath, "r", encoding="utf-8") as detail_file:
            for line in detail_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    loaded = cast(object, json.loads(line))
                    if not isinstance(loaded, dict):
                        continue
                    record = cast(dict[str, object], loaded)
                    filename = record.get("filename")
                    if isinstance(filename, str) and filename:
                        processed_files.add(filename)
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        print(f"[Warning] Failed to read existing progress from {filepath}: {exc}")

    return processed_files


def load_detail_results(filepath: str) -> list[dict[str, object]]:
    if not os.path.exists(filepath):
        return []

    results: list[dict[str, object]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as detail_file:
            for line in detail_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    loaded = cast(object, json.loads(line))
                    if not isinstance(loaded, dict):
                        continue
                    results.append(cast(dict[str, object], loaded))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        print(f"[Warning] Failed to load detail results from {filepath}: {exc}")
    return results


def resolve_original_image_path(image_path: str, input_path: str, original_dir: Optional[str]) -> Optional[str]:
    if not image_path:
        return None

    if os.path.isabs(image_path):
        return image_path if os.path.exists(image_path) else None

    candidates = [
        os.path.abspath(os.path.join(os.getcwd(), image_path)),
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(input_path)), image_path)),
    ]

    if original_dir:
        candidates.extend(
            [
                os.path.abspath(os.path.join(original_dir, image_path)),
                os.path.abspath(os.path.join(original_dir, os.path.basename(image_path))),
            ]
        )

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def process_single_entry(
    entry: dict[str, object],
    metric: object,
    edited_dir: str,
    input_path: str,
    original_dir: Optional[str],
) -> Optional[dict[str, object]]:
    try:
        filename_obj = entry.get("image_filename")
        if not isinstance(filename_obj, str) or not filename_obj:
            return None
        filename = filename_obj

        image_path_obj = entry.get("image_path")
        instruction_obj = entry.get("instruction")
        edit_type_obj = entry.get("type")
        expected_edit_word_count_obj = entry.get("expected_edit_word_count")

        if not isinstance(image_path_obj, str) or not image_path_obj:
            return None
        if not isinstance(instruction_obj, str) or not instruction_obj.strip():
            return None
        if not isinstance(edit_type_obj, str) or not edit_type_obj:
            return None

        edit_data = SimpleEditData(
            edit_type=edit_type_obj,
            instruction=instruction_obj.strip(),
            expected_edit_word_count=as_int(expected_edit_word_count_obj, 1),
        )

        edited_image_path = os.path.join(edited_dir, filename)
        original_image_path = resolve_original_image_path(image_path_obj, input_path, original_dir)

        if not os.path.exists(edited_image_path):
            print(f"[Skip] Edited image not found: {edited_image_path}")
            return None

        if not original_image_path or not os.path.exists(original_image_path):
            print(f"[Skip] Original image not found: {image_path_obj}")
            return None

        metric_result_obj = metric.calculate(
            original_image_path=original_image_path,
            edited_image_path=edited_image_path,
            edit_data=edit_data,
        )
        metric_result = as_dict(metric_result_obj)
        if not metric_result:
            print(f"[Error] Invalid metric result for {filename}")
            return None

        if "error" in metric_result:
            print(f"[Error] {filename}: {metric_result['error']}")
            return None

        answers = as_dict(metric_result.get("answers"))
        align_counts = as_dict(metric_result.get("align_counts"))
        issues_obj = metric_result.get("issues")
        issues: list[object] = cast(list[object], issues_obj) if isinstance(issues_obj, list) else []

        return {
            "filename": filename,
            "text_accuracy_score": metric_result.get("text_accuracy_score"),
            "visual_quality_score": metric_result.get("visual_quality_score"),
            "visual_subscores": as_dict(metric_result.get("visual_subscores")),
            "answers": answers,
            "align_counts": align_counts,
            "expected_edit_word_count": metric_result.get("expected_edit_word_count", as_int(expected_edit_word_count_obj, 1)),
            "edit_word_error_rate": metric_result.get("edit_word_error_rate"),
            "reason": metric_result.get("reason", ""),
            "issues": issues,
        }

    except Exception as exc:
        filename = entry.get("image_filename", "unknown")
        print(f"[Error] Processing {filename}: {exc}")
        return None


def aggregate_results(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        return {
            "average_total_word_errors": 0.0,
            "average_word_error_rate": 0.0,
            "average_text_accuracy_score": 0.0,
            "average_visual_quality_score": 0.0,
            "average_expected_edit_word_count": 0.0,
            "average_edit_word_error_rate": 0.0,
            "average_location_correct": 0.0,
            "average_style_consistent": 0.0,
            "average_physical_plausibility": 0.0,
            "total_samples": 0,
            "details": [],
        }

    total_samples = len(results)
    average_total_word_errors = (
        sum(as_float(as_dict(r.get("align_counts")).get("total_word_errors", 0.0)) for r in results)
        / total_samples
    )
    average_word_error_rate = (
        sum(as_float(as_dict(r.get("align_counts")).get("word_error_rate", 0.0)) for r in results)
        / total_samples
    )
    average_text_accuracy_score = sum(as_float(r.get("text_accuracy_score", 0.0)) for r in results) / total_samples
    average_visual_quality_score = sum(as_float(r.get("visual_quality_score", 0.0)) for r in results) / total_samples
    average_expected_edit_word_count = sum(as_float(r.get("expected_edit_word_count", 0.0)) for r in results) / total_samples
    average_edit_word_error_rate = sum(as_float(r.get("edit_word_error_rate", 0.0)) for r in results) / total_samples
    average_location_correct = (
        sum(as_float(as_dict(r.get("visual_subscores")).get("location_correct", 0.0)) for r in results)
        / total_samples
    )
    average_style_consistent = (
        sum(as_float(as_dict(r.get("visual_subscores")).get("style_consistent", 0.0)) for r in results)
        / total_samples
    )
    average_physical_plausibility = (
        sum(as_float(as_dict(r.get("visual_subscores")).get("physical_plausibility", 0.0)) for r in results)
        / total_samples
    )

    return {
        "average_total_word_errors": average_total_word_errors,
        "average_word_error_rate": average_word_error_rate,
        "average_text_accuracy_score": average_text_accuracy_score,
        "average_visual_quality_score": average_visual_quality_score,
        "average_expected_edit_word_count": average_expected_edit_word_count,
        "average_edit_word_error_rate": average_edit_word_error_rate,
        "average_location_correct": average_location_correct,
        "average_style_consistent": average_style_consistent,
        "average_physical_plausibility": average_physical_plausibility,
        "total_samples": total_samples,
        "details": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate TA/VQ scores for edited images (whole-image text alignment)",
    )
    parser.add_argument("--input", required=True, help="Path to flat instruction JSONL input")
    parser.add_argument("--edited-dir", required=True, help="Directory containing model-generated edited images")
    parser.add_argument("--output", required=True, help="Path to output summary JSON")
    parser.add_argument("--details", required=True, help="Path to detailed JSONL (resume support)")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--original-dir",
        default=None,
        help="Optional fallback directory for original images",
    )

    args = parser.parse_args()

    input_path = cast(str, args.input)
    edited_dir = cast(str, args.edited_dir)
    output_path = cast(str, args.output)
    details_path = cast(str, args.details)
    original_dir = cast(Optional[str], args.original_dir)
    max_workers = cast(int, args.max_workers)

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    if not os.path.isdir(edited_dir):
        print(f"Error: Edited image directory not found: {edited_dir}")
        sys.exit(1)
    if original_dir is not None and not os.path.isdir(original_dir):
        print(f"Error: Original image directory not found: {original_dir}")
        sys.exit(1)
    if max_workers < 1:
        print("Error: --max-workers must be >= 1")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(details_path)), exist_ok=True)

    processed_files = get_existing_progress(details_path)
    print(f"Found {len(processed_files)} already processed images.")

    jobs: list[dict[str, object]] = []
    try:
        with open(input_path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue

                try:
                    loaded = cast(object, json.loads(line))
                    if not isinstance(loaded, dict):
                        continue
                    entry = cast(dict[str, object], loaded)

                    filename_obj = entry.get("image_filename")
                    if not isinstance(filename_obj, str) or filename_obj in processed_files:
                        continue

                    image_path_obj = entry.get("image_path")
                    instruction_obj = entry.get("instruction")
                    edit_type_obj = entry.get("type")

                    if not isinstance(image_path_obj, str) or not image_path_obj:
                        continue
                    if not isinstance(instruction_obj, str) or not instruction_obj.strip():
                        continue
                    if not isinstance(edit_type_obj, str) or not edit_type_obj:
                        continue

                    if resolve_original_image_path(image_path_obj, input_path, original_dir) is None:
                        continue

                    jobs.append(entry)
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        print(f"Error: Failed to read input file {input_path}: {exc}")
        sys.exit(1)


    print(f"New tasks to process: {len(jobs)}")

    try:
        from metrics.ta_vq import TAVQMetric
    except ImportError as exc:
        print(f"Error: Unable to import TAVQMetric: {exc}")
        sys.exit(1)

    metric = TAVQMetric(max_workers=max_workers)

    if jobs:
        with open(details_path, "a", encoding="utf-8") as detail_file:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_single_entry,
                        job,
                        metric,
                        edited_dir,
                        input_path,
                        original_dir,
                    )
                    for job in jobs
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(jobs),
                    desc="Evaluating TA/VQ",
                ):
                    result = future.result()
                    if result is None:
                        continue

                    _ = detail_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    detail_file.flush()

    all_results = load_detail_results(details_path)
    summary = aggregate_results(all_results)

    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"Error: Failed to write summary output {output_path}: {exc}")
        sys.exit(1)

    print("\nEvaluation Complete!")
    print(f"Total Processed: {summary['total_samples']}")
    print(f"Average Total Word Errors: {summary['average_total_word_errors']:.4f}")
    print(f"Average Word Error Rate: {summary['average_word_error_rate']:.6f}")
    print(f"Average Text Accuracy Score: {summary['average_text_accuracy_score']:.6f}")
    print(f"Average Visual Quality Score: {summary['average_visual_quality_score']:.6f}")
    print(f"Average Expected Edit Word Count: {summary['average_expected_edit_word_count']:.4f}")
    print(f"Average Edit Word Error Rate: {summary['average_edit_word_error_rate']:.6f}")
    print(f"Average Location Correct: {summary['average_location_correct']:.6f}")
    print(f"Average Style Consistent: {summary['average_style_consistent']:.6f}")
    print(f"Average Physical Plausibility: {summary['average_physical_plausibility']:.6f}")
    print(f"Results saved to: {output_path}")
    print(f"Details saved to: {details_path}")


if __name__ == "__main__":
    main()
