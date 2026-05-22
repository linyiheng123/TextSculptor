#!/usr/bin/env python3

import os
import sys
import json
import cv2
import numpy as np
import argparse
from typing import Any, Dict, Optional, List

from paddleocr import PaddleOCR
from skimage.metrics import structural_similarity
from tqdm import tqdm

from metrics.utils import ensure_dir, load_jsonl_lines

DILATION_KERNEL_SIZE = 10


def detect_text_regions_ocr(img: np.ndarray, ocr_engine: PaddleOCR) -> List[List[int]]:
    try:
        result = ocr_engine.ocr(img, cls=True)
        
        if not result or not result[0]:
            return []
        
        ocr_data = result[0]
        bboxes = []
        
        h, w = img.shape[:2]
        
        for item in ocr_data:
            # item[0] is the quadrilateral points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            box_points = item[0]
            
            # Convert to axis-aligned bbox [x1, y1, x2, y2]
            xs = [p[0] for p in box_points]
            ys = [p[1] for p in box_points]
            
            x1 = max(0, int(min(xs)))
            y1 = max(0, int(min(ys)))
            x2 = min(w, int(max(xs)))
            y2 = min(h, int(max(ys)))
            
            # Only add if box has valid size
            if x2 > x1 and y2 > y1:
                bboxes.append([x1, y1, x2, y2])
        
        return bboxes
        
    except Exception as e:
        print(f"[Error] OCR detection failed: {e}")
        return []


def calculate_background_metrics(
    img_orig: np.ndarray,
    img_edit: np.ndarray,
    ignore_bboxes: List[List[int]]
) -> tuple:
    h, w = img_orig.shape[:2]

    mask_ignore = np.zeros((h, w), dtype=np.uint8)
    
    for bbox in ignore_bboxes:
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cv2.rectangle(mask_ignore, (x1, y1), (x2, y2), 255, -1)
    
    kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
    mask_ignore = cv2.dilate(mask_ignore, kernel, iterations=1)

    background_mask = (mask_ignore == 0)
    
    if np.sum(background_mask) == 0:
        return None, mask_ignore
    
    metrics = {}

    img1 = img_orig.astype(np.float64)
    img2 = img_edit.astype(np.float64)
    diff = img1 - img2
    sq_diff = diff ** 2
    mse_val = np.mean(sq_diff[background_mask])
    
    if mse_val == 0:
        psnr_val = 100.0
    else:
        psnr_val = 20 * np.log10(255.0 / np.sqrt(mse_val))
        psnr_val = min(psnr_val, 100.0)
    
    metrics["mse"] = float(mse_val)
    metrics["psnr"] = float(psnr_val)
    
    win_size = min(11, h, w)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3
    
    try:
        _, ssim_map = structural_similarity(
            img_orig, img_edit,
            win_size=win_size,
            channel_axis=2,
            full=True,
            data_range=255
        )
        ssim_bg = np.mean(ssim_map[background_mask])
        metrics["ssim"] = float(ssim_bg)
    except Exception as e:
        print(f"[SSIM Error] {e}")
        metrics["ssim"] = 0.0
    
    return metrics, mask_ignore


def visualize_and_save(
    img_orig: np.ndarray,
    img_edit: np.ndarray,
    mask_ignore: np.ndarray,
    metrics: Dict[str, float],
    save_path: str
):
    h, w = img_orig.shape[:2]
    if img_edit.shape[:2] != (h, w):
        img_edit = cv2.resize(img_edit, (w, h))
    
    def apply_mask_overlay(image, mask, color=(0, 0, 255), alpha=0.3):
        out = image.copy()
        region = (mask == 255)
        if not np.any(region):
            return out
        color_layer = np.zeros_like(image)
        color_layer[region] = color
        img_roi = out[region]
        color_roi = color_layer[region]
        blended_roi = cv2.addWeighted(img_roi, 1 - alpha, color_roi, alpha, 0)
        out[region] = blended_roi
        return out
    
    img_orig_vis = apply_mask_overlay(img_orig, mask_ignore)
    img_edit_vis = apply_mask_overlay(img_edit, mask_ignore)
    
    spacer = np.zeros((h, 10, 3), dtype=np.uint8)
    combined = np.hstack([img_orig_vis, spacer, img_edit_vis])

    header_h = 60
    header = np.zeros((header_h, combined.shape[1], 3), dtype=np.uint8)
    final_img = np.vstack([header, combined])
    
    text = f"MSE: {metrics['mse']:.4f} | PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f}"
    cv2.putText(final_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imwrite(save_path, final_img)


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


def get_existing_progress(details_path: str) -> set:
    processed = set()
    if os.path.exists(details_path):
        try:
            with open(details_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    filename = entry.get('filename')
                    if filename:
                        processed.add(filename)
            print(f"[Resume] Found {len(processed)} already processed samples in {details_path}")
        except Exception as e:
            print(f"[Warning] Failed to load existing progress: {e}")
    return processed


def evaluate_one(
    entry: Dict[str, Any],
    input_path: str,
    edited_img_dir: str,
    original_dir: Optional[str],
    ocr_engine: PaddleOCR,
    visualize: bool = False,
    vis_dir: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    try:
        filename = entry.get('image_filename')
        original_path_raw = entry.get('image_path')
        edit_type = entry.get('type')

        if not isinstance(filename, str) or not filename:
            return None
        if not isinstance(original_path_raw, str) or not original_path_raw:
            return None
        if not isinstance(edit_type, str) or not edit_type:
            return None

        original_path = resolve_original_image_path(original_path_raw, input_path, original_dir)
        edited_path = os.path.join(edited_img_dir, filename)

        if not original_path or not os.path.exists(original_path):
            return None
        if not os.path.exists(edited_path):
            return None

        img_orig = cv2.imread(original_path)
        img_edit = cv2.imread(edited_path)

        if img_orig is None or img_edit is None:
            return None

        h, w = img_orig.shape[:2]
        if img_edit.shape[:2] != (h, w):
            print(f"[Warning] Size mismatch for {filename}: {img_edit.shape[:2]} != {img_orig.shape[:2]}")
            img_edit = cv2.resize(img_edit, (w, h))

        bboxes_orig = detect_text_regions_ocr(img_orig, ocr_engine)
        bboxes_edit = detect_text_regions_ocr(img_edit, ocr_engine)

        all_bboxes = bboxes_orig + bboxes_edit
        metrics, mask_ignore = calculate_background_metrics(img_orig, img_edit, all_bboxes)
        
        if metrics is None:
            return None

        if visualize and vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
            vis_name = f"{os.path.splitext(filename)[0]}_bp_vis.jpg"
            save_path = os.path.join(vis_dir, vis_name)
            visualize_and_save(img_orig, img_edit, mask_ignore, metrics, save_path)
        
        return {
            'filename': filename,
            'edit_type': edit_type,
            'mse': metrics['mse'],
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim'],
        }
        
    except Exception as e:
        filename = entry.get('image_filename', 'unknown')
        print(f"[Error] Processing {filename}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Background Preservation metrics (MSE, PSNR, SSIM) using OCR-based approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cal_bp.py --input replace.jsonl --edited-dir output/seedream4p0/replace --output results/bp.json --details results/bp_details.jsonl
  
  # With visualization
  python cal_bp.py --input instructions.jsonl --edited-dir output/images --output results/bp.json --details results/bp_details.jsonl --visualize --vis-dir vis/bp
        """
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input instruction JSONL file"
    )
    parser.add_argument(
        "--edited-dir",
        required=True,
        help="Directory containing model-generated edited images"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file for aggregated metrics"
    )
    parser.add_argument(
        "--details",
        required=True,
        help="Output JSONL file for detailed results (supports resume)"
    )
    parser.add_argument(
        "--original-dir",
        default=None,
        help="Optional fallback directory for original images"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization images with mask overlay"
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="vis/bp",
        help="Directory for visualization output (default: vis/bp)"
    )
    
    args = parser.parse_args()
    
    print(f"Background Preservation Metric Calculator")
    print(f"Input: {args.input}")
    print(f"Edited Dir: {args.edited_dir}")
    print(f"Output: {args.output}")
    print(f"Details: {args.details}")
    print(f"Original Dir: {args.original_dir}")
    print(f"Visualize: {args.visualize}")
    if args.visualize:
        print(f"Vis Dir: {args.vis_dir}")
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    if not os.path.isdir(args.edited_dir):
        print(f"Error: Edited image directory not found: {args.edited_dir}")
        sys.exit(1)
    if args.original_dir is not None and not os.path.isdir(args.original_dir):
        print(f"Error: Original image directory not found: {args.original_dir}")
        sys.exit(1)

    processed_filenames = get_existing_progress(args.details)
    all_entries = load_jsonl_lines(args.input)
    print(f"\nLoaded {len(all_entries)} entries from {args.input}")

    entries_to_process = [
        entry for entry in all_entries
        if entry.get('image_filename') not in processed_filenames
    ]
    
    skipped_count = len(all_entries) - len(entries_to_process)
    if skipped_count > 0:
        print(f"[Resume] Skipping {skipped_count} already processed samples")
    print(f"Processing {len(entries_to_process)} samples...")
    
    if len(entries_to_process) == 0:
        print("[Done] All samples already processed!")
    else:
        ensure_dir(os.path.dirname(args.output))
        ensure_dir(os.path.dirname(args.details))
        
        if args.visualize:
            os.makedirs(args.vis_dir, exist_ok=True)

        print("\nInitializing PaddleOCR...")
        ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
        )
        print("PaddleOCR loaded successfully")

        success_count = 0
        
        for entry in tqdm(entries_to_process, desc="Evaluating"):
            try:
                result = evaluate_one(
                    entry,
                    args.input,
                    args.edited_dir,
                    args.original_dir,
                    ocr_engine,
                    args.visualize,
                    args.vis_dir,
                )
                if result:
                    with open(args.details, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    success_count += 1
            except Exception as e:
                print(f"\n[Error] {e}")
        
        print(f"\n[Done] Successfully processed {success_count} samples")
    
    print("\nCalculating aggregate statistics...")
    
    if not os.path.exists(args.details):
        print("[Warning] No details file found, cannot calculate aggregate stats")
        return
    
    all_results = []
    with open(args.details, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                all_results.append(result)
            except:
                pass
    
    if len(all_results) == 0:
        print("[Warning] No valid results in details file")
        return
    
    total_mse = sum(r['mse'] for r in all_results)
    total_psnr = sum(r['psnr'] for r in all_results)
    total_ssim = sum(r['ssim'] for r in all_results)
    n_total = len(all_results)
    
    by_type = {}
    for r in all_results:
        edit_type = r.get('edit_type', 'unknown')
        if edit_type not in by_type:
            by_type[edit_type] = []
        by_type[edit_type].append(r)
    
    edit_type_breakdown = {}
    for edit_type, samples in by_type.items():
        mse_vals = [s['mse'] for s in samples]
        psnr_vals = [s['psnr'] for s in samples]
        ssim_vals = [s['ssim'] for s in samples]
        
        edit_type_breakdown[edit_type] = {
            'count': len(samples),
            'avg_mse': round(sum(mse_vals) / len(mse_vals), 4),
            'avg_psnr': round(sum(psnr_vals) / len(psnr_vals), 3),
            'avg_ssim': round(sum(ssim_vals) / len(ssim_vals), 4),
        }
    
    output_data = {
        'average_bp_score': round(total_ssim / n_total, 4),
        'avg_mse': round(total_mse / n_total, 4),
        'avg_psnr': round(total_psnr / n_total, 3),
        'avg_ssim': round(total_ssim / n_total, 4),
        'total_samples': n_total,
        'edit_type_breakdown': edit_type_breakdown,
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Background Preservation Results")
    print(f"{'='*60}")
    print(f"Overall Statistics:")
    print(f"  Total samples: {n_total}")
    print(f"  Avg MSE: {output_data['avg_mse']:.4f}")
    print(f"  Avg PSNR: {output_data['avg_psnr']:.3f}")
    print(f"  Avg SSIM: {output_data['avg_ssim']:.4f}")
    print(f"\nBreakdown by Edit Type:")
    for edit_type, stats in edit_type_breakdown.items():
        print(f"  {edit_type}: {stats['count']} samples")
        print(f"    MSE={stats['avg_mse']:.4f}, PSNR={stats['avg_psnr']:.3f}, SSIM={stats['avg_ssim']:.4f}")
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"Details saved to: {args.details}")
    if args.visualize:
        print(f"Visualizations saved to: {args.vis_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
