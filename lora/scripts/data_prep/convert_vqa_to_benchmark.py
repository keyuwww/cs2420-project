#!/usr/bin/env python3
"""
Convert VQA subset to benchmark format.
Extracts questions as safe prompts with label=0.

Output format:
{"image_path": "image/filename.jpg", "prompt": "question text", "label": 0}
"""

import json
import os
from pathlib import Path

def convert_vqa_to_benchmark(vqa_json_path, output_path, image_dir_for_validation=None):
    """
    Convert VQA subset.json to benchmark format.

    Args:
        vqa_json_path: Path to VQA subset.json
        output_path: Path to output benchmark_safe.json
        image_dir_for_validation: Optional path to validate image existence
    """
    print(f"Reading VQA data from: {vqa_json_path}")

    with open(vqa_json_path, 'r', encoding='utf-8') as f:
        vqa_data = json.load(f)

    print(f"Loaded {len(vqa_data)} image entries")

    benchmark_entries = []
    images_found = 0
    images_missing = 0
    total_questions = 0

    for entry in vqa_data:
        image_id = entry['image_id']
        image_file = entry.get('image_file', f'images/{image_id}.jpg')

        # Extract just the filename and convert to image/filename.jpg format
        filename = os.path.basename(image_file)
        image_path = f"image/{filename}"

        # Validate image exists if validation directory provided
        if image_dir_for_validation:
            full_path = os.path.join(image_dir_for_validation, image_path)
            if not os.path.exists(full_path):
                images_missing += 1
                print(f"Warning: Image not found: {image_path}")
                continue
            else:
                images_found += 1

        # Extract all questions for this image
        questions = entry.get('questions', [])

        for q in questions:
            question_text = q.get('question', '').strip()

            if question_text:
                total_questions += 1
                benchmark_entries.append({
                    'image_path': image_path,
                    'prompt': question_text,
                    'label': 0  # VQA questions are safe
                })

    # Write to output file (JSONL format)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in benchmark_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\nConversion complete!")
    print(f"  Total questions extracted: {total_questions}")
    print(f"  Images found: {images_found}")
    print(f"  Images missing: {images_missing}")
    print(f"  Output saved to: {output_path}")

    return benchmark_entries

if __name__ == "__main__":
    import sys

    # Go up to lora/ directory
    lora_dir = Path(__file__).parent.parent.parent
    data_dir = lora_dir / "data"

    # Default paths
    vqa_subset_dir = lora_dir / "VQA_subset_1602"
    vqa_json = vqa_subset_dir / "subset.json"
    output_file = data_dir / "benchmark_safe.json"

    # Optional: validate against images in lora/data/image
    # Set to None to skip validation (useful before copying images)
    image_validation_dir = None  # data_dir

    if len(sys.argv) > 1:
        vqa_json = Path(sys.argv[1])

    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])

    if not vqa_json.exists():
        print(f"Error: VQA subset.json not found at: {vqa_json}")
        print(f"\nUsage: python {Path(__file__).name} [vqa_subset.json] [output_file.json]")
        sys.exit(1)

    # Convert
    convert_vqa_to_benchmark(vqa_json, output_file, image_validation_dir)

    print(f"\n✓ Done! Safe benchmark created.")
    print(f"\nNext steps:")
    print(f"1. Copy VQA images to lora/data/image/:")
    print(f"   cp {vqa_subset_dir}/images/* {script_dir / 'data' / 'image'}/")
    print(f"2. Combine with unsafe benchmark if needed:")
    print(f"   cat data/benchmark.json data/benchmark_safe.json > data/benchmark_combined.json")
